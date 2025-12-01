// C++ 实现的 Temporal-SARL 挖掘器，使用 LibTorch + OpenMP
// 功能：加载 TorchScript 模型，读取时序三元组，双向挖掘路径，Top-K 采样，输出 sarl_raw_paths.txt

#include <torch/torch.h>
#include <torch/script.h>

#include <omp.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

struct TemporalNeighbor {
    int dst;
    int relation;
    double timestamp;
};

struct Triple {
    int head;
    int relation;
    int tail;
    double ts;
};

struct Options {
    int max_hops = 3;
    int history_size = 8;
    int beam_size = 16;
    double time_window = 30 * 86400.0;
    bool use_cuda = false;
    int topk = 5;
    std::string log_file = "sarl_raw_paths.txt";
};

// 读取简单的 json 映射 {"name": id, ...}
std::map<std::string, int> load_map(const std::string &path) {
    std::ifstream fin(path);
    if (!fin) throw std::runtime_error("cannot open map file: " + path);
    std::string content((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
    std::map<std::string, int> m;
    std::string key;
    int val = 0;
    bool in_key = false;
    std::string token;
    for (size_t i = 0; i < content.size(); ++i) {
        char c = content[i];
        if (c == '\"') {
            if (!in_key) { key.clear(); in_key = true; }
            else { in_key = false; }
        } else if (in_key) {
            key.push_back(c);
        } else if (c >= '0' && c <= '9') {
            token.push_back(c);
        } else {
            if (!key.empty() && !token.empty()) {
                val = std::stoi(token);
                m[key] = val;
                key.clear();
                token.clear();
            } else {
                token.clear();
            }
        }
    }
    if (!key.empty() && !token.empty()) {
        val = std::stoi(token);
        m[key] = val;
    }
    return m;
}

// 解析 train/valid/test.txt : head \t rel \t tail \t date(YYYY-MM-DD)
std::vector<Triple> load_triples(const std::string &dataset_dir) {
    std::vector<Triple> triples;
    for (auto split : {"train.txt", "valid.txt", "test.txt"}) {
        std::ifstream fin(dataset_dir + "/" + split);
        if (!fin) continue;
        std::string line;
        while (std::getline(fin, line)) {
            std::stringstream ss(line);
            std::string h, r, t, date;
            if (!std::getline(ss, h, '\t')) continue;
            if (!std::getline(ss, r, '\t')) continue;
            if (!std::getline(ss, t, '\t')) continue;
            if (!std::getline(ss, date, '\t')) continue;
            // 转换日期为时间戳（粗略：按天）
            int y = 0, m = 0, d = 0;
            if (sscanf(date.c_str(), "%d-%d-%d", &y, &m, &d) != 3) continue;
            std::tm tm{};
            tm.tm_year = y - 1900;
            tm.tm_mon = m - 1;
            tm.tm_mday = d;
            double ts = std::mktime(&tm);
            triples.push_back({std::stoi(h), std::stoi(r), std::stoi(t), ts});
        }
    }
    return triples;
}

using Adj = std::map<int, std::vector<TemporalNeighbor>>;

Adj build_adj(const std::vector<Triple> &triples, bool reverse = false) {
    Adj adj;
    for (auto &tr : triples) {
        int src = reverse ? tr.tail : tr.head;
        int dst = reverse ? tr.head : tr.tail;
        adj[src].push_back({dst, tr.relation, tr.ts});
    }
    for (auto &kv : adj) {
        auto &vec = kv.second;
        std::sort(vec.begin(), vec.end(), [](auto &a, auto &b) { return a.timestamp > b.timestamp; });
    }
    return adj;
}

struct Miner {
    torch::jit::script::Module model;
    Options opt;
    Adj adj;
    Adj rev_adj;
    int history_size;
    int topk;
    std::mutex log_mutex;
    std::ofstream log_stream;
    torch::Device device;

    Miner(const std::string &model_path, const Options &o, const Adj &a, const Adj &ra)
        : opt(o), adj(a), rev_adj(ra), history_size(o.history_size), topk(o.topk),
          device(o.use_cuda ? torch::kCUDA : torch::kCPU) {
        model = torch::jit::load(model_path, device);
        model.eval();
        log_stream.open(opt.log_file, std::ios::out);
        if (!log_stream) throw std::runtime_error("cannot open log file");
    }

    // 获取时间窗内邻居，截断 beam_size
    std::vector<TemporalNeighbor> get_neighbors(int node, double current_time, bool reverse) {
        const Adj &g = reverse ? rev_adj : adj;
        auto it = g.find(node);
        std::vector<TemporalNeighbor> res;
        if (it == g.end()) return res;
        double lower = current_time - opt.time_window;
        for (auto &e : it->second) {
            if (e.timestamp > current_time || e.timestamp < lower) continue;
            res.push_back(e);
            if ((int)res.size() >= opt.beam_size) break;
        }
        return res;
    }

    // 构建历史张量
    void build_history(torch::Tensor &h_ent, torch::Tensor &h_rel, torch::Tensor &h_dt,
                       const std::vector<int> &ents, const std::vector<int> &rels,
                       const std::vector<double> &dts) {
        auto opts_l = torch::TensorOptions().dtype(torch::kLong).device(device);
        auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        h_ent = torch::from_blob((void*)ents.data(), {(int64_t)1, (int64_t)history_size}, opts_l).clone();
        h_rel = torch::from_blob((void*)rels.data(), {(int64_t)1, (int64_t)history_size}, opts_l).clone();
        h_dt  = torch::from_blob((void*)dts.data(),  {(int64_t)1, (int64_t)history_size}, opts_f).clone();
    }

    // 选择邻居（Top-K + multinomial）
    TemporalNeighbor select_neighbor(const std::vector<TemporalNeighbor> &cands,
                                     std::vector<int> &hist_ent,
                                     std::vector<int> &hist_rel,
                                     std::vector<double> &hist_dt,
                                     int current_entity, int relation_id, double query_time) {
        int num_cand = (int)cands.size();
        auto opts_l = torch::TensorOptions().dtype(torch::kLong).device(device);
        auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        torch::Tensor cand_entities = torch::zeros({1, num_cand}, opts_l);
        torch::Tensor cand_relations = torch::zeros({1, num_cand}, opts_l);
        torch::Tensor cand_deltas = torch::zeros({1, num_cand}, opts_f);
        for (int i = 0; i < num_cand; ++i) {
            cand_entities[0][i] = cands[i].dst;
            cand_relations[0][i] = cands[i].relation;
            cand_deltas[0][i] = std::max(0.0, query_time - cands[i].timestamp);
        }
        torch::Tensor h_ent, h_rel, h_dt;
        build_history(h_ent, h_rel, h_dt, hist_ent, hist_rel, hist_dt);
        torch::Tensor cur_ent = torch::tensor({current_entity}, opts_l);
        torch::Tensor query_rel = torch::tensor({relation_id}, opts_l);

        std::vector<torch::jit::IValue> inputs{
            h_ent, h_rel, h_dt, cur_ent, query_rel, cand_entities, cand_relations, cand_deltas};

        torch::Tensor scores = model.forward(inputs).toTensor();
        // 将 logits 拉回 CPU 再做后续处理，避免多线程 GPU 采样开销
        scores = scores.to(torch::kCPU);
        scores = torch::nan_to_num(scores, 0.0, 0.0, 0.0);
        auto probs = torch::softmax(scores, -1);
        if (!probs.isfinite().all().item<bool>() || probs.sum().item<double>() <= 0) {
            probs = torch::ones_like(probs) / (double)num_cand;
        }
        int k = std::min(topk, num_cand);
        auto top = std::get<0>(torch::topk(probs, k));
        auto idx = std::get<1>(torch::topk(probs, k));
        auto norm = torch::softmax(top, -1);
        norm = torch::nan_to_num(norm, 0.0, 0.0, 0.0);
        if (!norm.isfinite().all().item<bool>() || norm.sum().item<double>() <= 0) {
            norm = torch::ones_like(norm) / (double)k;
        }
        auto sampled = torch::multinomial(norm, 1).item<int>();
        int chosen = idx[sampled].item<int>();
        return cands[chosen];
    }

    // 单次游走
    bool single_walk(int pivot, int relation_id, double query_time, bool reverse,
                     std::vector<TemporalNeighbor> &out_edges) {
        std::vector<int> hist_ent(history_size, 0);
        std::vector<int> hist_rel(history_size, 0);
        std::vector<double> hist_dt(history_size, 0.0);
        hist_ent[0] = pivot;
        hist_rel[0] = relation_id;
        int current = pivot;
        double current_time = query_time;
        out_edges.clear();
        for (int hop = 0; hop < opt.max_hops; ++hop) {
            auto neighbors = get_neighbors(current, current_time, reverse);
            if (neighbors.empty()) return false;
            auto choice = select_neighbor(neighbors, hist_ent, hist_rel, hist_dt, current, relation_id, query_time);
            out_edges.push_back(choice);
            // 更新历史（FIFO）
            hist_ent.push_back(choice.dst);
            hist_rel.push_back(choice.relation);
            hist_dt.push_back(std::max(0.0, query_time - choice.timestamp));
            if ((int)hist_ent.size() > history_size) {
                hist_ent.erase(hist_ent.begin());
                hist_rel.erase(hist_rel.begin());
                hist_dt.erase(hist_dt.begin());
            }
            current = choice.dst;
            current_time = choice.timestamp;
        }
        return !out_edges.empty();
    }

    // 写路径日志
    void write_path(int pivot, int relation_id, double query_time,
                    const std::vector<TemporalNeighbor> &edges, const std::string &side) {
        std::lock_guard<std::mutex> guard(log_mutex);
        log_stream << "Query[" << side << "](" << pivot << ", " << relation_id
                   << " @ " << (long long)query_time << ") -> Path: ";
        for (size_t i = 0; i < edges.size(); ++i) {
            log_stream << edges[i].relation << "(" << (long long)edges[i].timestamp
                       << ") => " << edges[i].dst;
            if (i + 1 < edges.size()) log_stream << " -> ";
        }
        log_stream << "\n";
    }
};

int main(int argc, char *argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: ./sarl_miner <dataset_dir> <model_pt> <num_queries> <walks_per_query> <output_raw_paths>\n";
        return 1;
    }
    std::string dataset_dir = argv[1];
    std::string model_pt = argv[2];
    int num_queries = std::stoi(argv[3]);
    int walks_per_query = std::stoi(argv[4]);
    std::string raw_out = argv[5];

    Options opt;
    opt.log_file = raw_out;
    opt.use_cuda = torch::cuda::is_available();
    std::cout << "[Device] CUDA available: " << std::boolalpha << opt.use_cuda << std::endl;

    // 加载数据
    auto triples = load_triples(dataset_dir);
    if (triples.empty()) {
        std::cerr << "No triples found\n";
        return 1;
    }
    std::cout << "[Init] loaded triples: " << triples.size() << std::endl;

    auto adj = build_adj(triples, false);
    auto rev = build_adj(triples, true);

    Miner miner(model_pt, opt, adj, rev);

    // 简单采样前 num_queries 个作为查询
    if (num_queries > (int)triples.size()) num_queries = (int)triples.size();

    auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_queries; ++i) {
        auto &q = triples[i];
        std::mt19937 rng((unsigned)(q.head * 131 + q.tail + omp_get_thread_num()));
        for (int w = 0; w < walks_per_query; ++w) {
            std::vector<TemporalNeighbor> edges;
            if (miner.single_walk(q.head, q.relation, q.ts, false, edges)) {
                miner.write_path(q.head, q.relation, q.ts, edges, "head");
            }
            if (miner.single_walk(q.tail, q.relation, q.ts, true, edges)) {
                miner.write_path(q.tail, q.relation, q.ts, edges, "tail");
            }
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[Done] elapsed " << sec / 60.0 << " min\n";
    return 0;
}

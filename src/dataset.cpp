#include "dataset.h"
#include "rng.h"

#include <fstream>
#include <iostream>

bool ByteDataset::load_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Failed to open dataset: " << path << "\n";
        return false;
    }
    data_.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    if (data_.size() < 2) {
        std::cerr << "Dataset too small\n";
        return false;
    }
    std::cerr << "Loaded " << data_.size() << " bytes\n";
    return true;
}

void ByteDataset::sample_batch(int batch, int seq_len, uint64_t seed,
                               std::vector<std::vector<uint8_t>>& inputs,
                               std::vector<std::vector<uint8_t>>& targets) const {
    inputs.assign(batch, std::vector<uint8_t>(seq_len));
    targets.assign(batch, std::vector<uint8_t>(seq_len));

    uint64_t st = seed;
    for (int b = 0; b < batch; ++b) {
        uint64_t r = rng::splitmix64(st);
        size_t start = static_cast<size_t>(r % (data_.size() - seq_len - 1));
        for (int t = 0; t < seq_len; ++t) {
            inputs[b][t] = data_[start + t];
            targets[b][t] = data_[start + t + 1];
        }
    }
}


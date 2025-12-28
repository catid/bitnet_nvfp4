#pragma once

#include <cstdint>
#include <string>
#include <vector>

class ByteDataset {
public:
    bool load_file(const std::string& path);

    // Sample `batch` sequences of length `seq_len` (inputs) + 1 (targets).
    void sample_batch(int batch, int seq_len, uint64_t seed,
                      std::vector<std::vector<uint8_t>>& inputs,
                      std::vector<std::vector<uint8_t>>& targets) const;

    size_t size() const { return data_.size(); }

private:
    std::vector<uint8_t> data_;
};


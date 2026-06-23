/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "amd_smi/impl/amd_smi_addc.h"

#include "amd_smi/impl/third_party/nlohmann/json.hpp"

namespace amdsmi {
namespace addc {

bool parse_afids_from_summary(const std::string& summary_json, std::vector<uint64_t>& out) {
  out.clear();

  nlohmann::json doc = nlohmann::json::parse(summary_json, nullptr, /*allow_exceptions=*/false);
  if (doc.is_discarded() || !doc.is_array()) {
    return false;
  }

  for (const auto& entry : doc) {
    auto it = entry.find("afid");
    if (it != entry.end() && it->is_number_unsigned()) {
      out.push_back(it->get<uint64_t>());
    }
  }
  return true;
}

}  // namespace addc
}  // namespace amdsmi

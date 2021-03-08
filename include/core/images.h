#pragma once
#include <vector>

namespace naivebayes {
typedef std::vector<std::vector<char>> Image;

class Images {
 public:
  /**
   * Overloads the >> operator to store each character into training_images_.
   *
   * @param is
   * @param data
   * @return given istream
   */
  friend std::istream& operator>>(std::istream& is, Images& data);

  std::vector<Image> GetImages() const;

  char GetPixel(const size_t image_index, const size_t row, const size_t col) const;

  size_t GetShade(const size_t image_index, const size_t row, const size_t col) const;

 private:
  std::vector<Image> images_;

};
}  // namespace naivebayes
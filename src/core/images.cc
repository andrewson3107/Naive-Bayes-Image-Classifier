#include <core/images.h>

#include <string>

namespace naivebayes {

std::istream& operator>>(std::istream& is, Images& data) {
  std::string line;
  // A 2d vector representing an n x n ascii image.
  std::vector<std::vector<char>> image;

  if (is.fail()) {
    throw std::invalid_argument("File does not exist or is blank");
  }

  while (getline(is, line)) {
    // Creates a vector of characters from the read line
    std::vector<char> line_char(line.begin(), line.end());
    image.push_back(line_char);

    // Every image is square, so if # of rows == # of characters in column,
    // image is complete and the next line will start the next image.
    if (image.size() == line.length()) {
      data.images_.push_back(image);
      image.clear();
    }
  }
  return is;
}

std::vector<Image> Images::GetImages() const {
  return images_;
}

char Images::GetPixel(const size_t image_index, const size_t row,
                      const size_t col) const {
  return images_.at(image_index).at(row).at(col);
}

size_t Images::GetShade(const size_t image_index, const size_t row,
                        const size_t col) const {
  if (images_.at(image_index).at(row).at(col) == ' ') {
    return 0;
  } else {
    return 1;
  }
}
}  // namespace naivebayes
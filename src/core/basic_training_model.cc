#include <core/basic_training_model.h>
#include <algorithm>
#include <fstream>
#include <iostream>

namespace naivebayes {

void BasicTrainingModel::SetImages(const Images& data_to_add) {
  training_images_ = data_to_add;
  image_size_ = training_images_.GetImages()[0].size();
}

std::istream& operator>>(std::istream& is, BasicTrainingModel& model) {
  if (is.fail()) {
    throw std::invalid_argument("File does not exist or is blank");
  }

  is >> model.image_size_;
  is >> model.num_classes_;

  for (size_t index = 0; index < model.num_classes_; index++) {
    size_t class_value;
    is >> class_value;
    model.classes_.push_back(class_value);
  }

  for (size_t class_num: model.classes_) {
    is >> model.class_probabilities_[class_num];
  }

  // Nested for loops iterate through every possible probability stored.
  for (size_t class_num: model.classes_) {
    std::vector<std::vector<double>> temp;
    for (size_t row = 0; row < model.image_size_; row++) {
      std::vector<double> probabilities;
      for (size_t col = 0; col < model.image_size_; col++) {
        double pixel_probability;
        is >> pixel_probability;
        probabilities.push_back(pixel_probability);
      }
      temp.push_back(probabilities);
    }
    model.pixel_probabilities_[class_num] = temp;
  }
  return is;
}

std::ostream& operator<<(std::ostream& os, const BasicTrainingModel& model) {
  // First writes the image size and number of classes.
  os << model.image_size_ << std::endl
     << model.num_classes_ << std::endl;

  for (size_t class_number: model.classes_) {
    os << class_number << std::endl;
  }

  // Then writes class probabilities.
  for (std::pair<const size_t, double> probability :
       model.class_probabilities_) {
    os << probability.second << std::endl;
  }

  // Nested for loops iterate through every possible probability stored.
  for (size_t class_number: model.classes_) {
    for (size_t row = 0; row < model.image_size_; row++) {
      for (size_t col = 0; col < model.image_size_; col++) {
        os << model.pixel_probabilities_.at(class_number).at(row).at(col)
           << std::endl;
      }
    }
  }
  return os;
}

void BasicTrainingModel::ReadLabels(const std::string& file_path) {
  size_t label;
  std::ifstream is(file_path);

  if (is.fail()) {
    throw std::invalid_argument("File does not exist or is blank");
  }

  while (is >> label) {
    image_labels_.push_back(label);
  }

  CountSizeNumClasses();
}

void BasicTrainingModel::CountSizeNumClasses() {
  // Creates a copy of image_classes and saves the number of unique elements as number
  // of classes.
  std::vector<size_t> temp(image_labels_);
  std::sort(temp.begin(), temp.end());
  num_classes_ = std::unique(temp.begin(), temp.end()) - temp.begin();

  // Stores one instance of each class.
  std::vector<size_t> copy(image_labels_);
  sort(copy.begin(),copy.end());
  copy.erase(unique(copy.begin(),copy.end() ),copy.end());
  classes_ = copy;

  // Counts the size of each class.
  for (size_t class_num: classes_) {
    size_t class_count =
        std::count(image_labels_.begin(), image_labels_.end(), class_num);
    class_sizes_[class_num] = class_count;
  }

}

void BasicTrainingModel::TrainModel() {
  CalculateClassProbability();
  CalculatePixelProbability();
}

void BasicTrainingModel::CalculateClassProbability() {
  for (size_t class_number: classes_) {

    double class_probability =
        (kLaplaceSmoothingValue + class_sizes_[class_number]) /
        (kLaplaceSmoothingValue * num_classes_ + image_labels_.size());
    class_probabilities_[class_number] = class_probability;
  }
}

void BasicTrainingModel::CalculatePixelProbability() {
  // Iterates through every possiblity.
  for (size_t class_number: classes_) {
    size_t image_count = 0;
    std::vector<std::vector<double>> temp;
    for (size_t row = 0; row < image_size_; row++) {
      std::vector<double> probabilities;
      for (size_t col = 0; col < image_size_; col++) {
        for (size_t index = 0; index < image_labels_.size(); index++) {
          // Iterates through image_classes and only continues if the class is the
          // given one.
          if (image_labels_[index] == class_number &&
              training_images_.GetPixel(index, row, col) == ' ') {
            // Number of images_ satisfying F(i,j) = ''.
            image_count++;
          }
        }
        double pixel_probability = (kLaplaceSmoothingValue + image_count) /
                                   (kLaplaceSmoothingValue * 2 + class_sizes_[class_number]);
        probabilities.push_back(pixel_probability);
        image_count = 0;
      }
      temp.push_back(probabilities);
    }
    pixel_probabilities_[class_number] = temp;
  }
}

double BasicTrainingModel::GetClassProbability(
    const size_t class_number) const {
  return class_probabilities_.at(class_number);
}

double BasicTrainingModel::GetPixelProbability(const size_t class_number,
                                               const size_t shade,
                                               const size_t row,
                                               const size_t col) const {
  if (shade == kShaded) {
    return 1 - pixel_probabilities_.at(class_number).at(row).at(col);
  } else {
    return pixel_probabilities_.at(class_number).at(row).at(col);
  }
}

std::vector<double> BasicTrainingModel::GetProbabilities() const {
  std::vector<double> temp;

  for (std::pair<size_t, double> probability : class_probabilities_) {
    temp.push_back(probability.second);
  }

  for (size_t class_num: classes_) {
    for (size_t row = 0; row < image_size_; row++) {
      for (size_t col = 0; col < image_size_; col++) {
        temp.push_back(pixel_probabilities_.at(class_num).at(row).at(col));
      }
    }
  }

  return temp;
}

std::vector<size_t> BasicTrainingModel::GetLabels() const {
  return image_labels_;
}

}  // namespace naivebayes
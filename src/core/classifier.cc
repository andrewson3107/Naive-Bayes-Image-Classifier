#include <core/classifier.h>

#include <fstream>

namespace naivebayes {

size_t Classifier::ClassifyImage(const Image& image) {
  size_t predicted_class = 0;
  double temp = -DBL_MAX;

  for (size_t class_num : model_.classes_) {
    double likelihood_score = CalculateLikelihoodScore(class_num, image);

    if (temp < likelihood_score) {
      temp = likelihood_score;
      predicted_class = class_num;
    }
  }

  return predicted_class;
}

double Classifier::CalculateLikelihoodScore(const size_t class_num, const Image image) {
  double likelihood_score = 0;

  likelihood_score += log10(model_.GetClassProbability(class_num));

  for (size_t row = 0; row < image.size(); row++) {
    for (size_t col = 0; col < image.size(); col++) {
      size_t shade;
      if (image[row][col] == ' ') {
        shade = kUnshaded;
      } else {
        shade = kShaded;
      }

      likelihood_score +=
          log10(model_.GetPixelProbability(class_num, shade, row, col));
    }
  }

  return likelihood_score;
}

double Classifier::CalculateAccuracy(const Images& images_to_classify) {
  std::vector<Image> images = images_to_classify.GetImages();
  size_t correct_count = 0;

  for (size_t index = 0; index < expected_class_.size(); index++) {
    if (ClassifyImage(images[index]) == expected_class_[index]) {
      correct_count++;
    }
  }

  double accuracy = ((double) correct_count) / (expected_class_.size());
  return accuracy;
}

void Classifier::ReadLabels(const std::string& file_path) {
  size_t label;
  std::ifstream is(file_path);

  if (is.fail()) {
    throw std::invalid_argument("File does not exist or is blank");
  }

  while (is >> label) {
    expected_class_.push_back(label);
  }
}
void Classifier::SetModel(BasicTrainingModel model) {
  model_ = model;
}

}  // namespace naivebayes

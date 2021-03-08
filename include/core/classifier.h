#pragma once
#include <core/basic_training_model.h>

namespace naivebayes {

class Classifier {
 public:
  BasicTrainingModel model_;
  Images images_;

  /**
   * Calculates the likelihood score of an image belonging to a class
   *
   * @param class_num The desired class of the likelihood score
   * @param image The image to classify
   * @return The likelihood score that the imagge belongs to class_num
   */
  double CalculateLikelihoodScore(const size_t class_num, const Image image);

  /**
   * Calculates the likelihood score of an image belonging to every class
   * possible. Determines which class has the highest likelihood score and
   * classifies the image as that class.
   *
   * @param image The image to classify
   * @return The class with the highest likelihood score
   */
  size_t ClassifyImage(const Image& image);

  void SetModel(BasicTrainingModel model);

  /**
   * Reads in the expected classes of each image that is used for testing
   * classifier accuracy.
   *
   * @param file_path file path of the labels file
   */
  void ReadLabels(const std::string& file_path);

  /**
   * Classifies every image within the images reference passed. Compares each
   * classification to the correct one within the label file. Returns a proportion
   * of correct images classified.
   * @param images_to_classify Images object that is a vector of images
   * @return a decimal representing the percent correctly classified.
   */
  double CalculateAccuracy(const Images& images_to_classify);

 private:
  static const size_t kShaded = 1;
  static const size_t kUnshaded = 0;
  std::vector<size_t> expected_class_;
};
}  // namespace naivebayes
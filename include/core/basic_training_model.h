#pragma once
#pragma warning(disable : 4503)
#include <core/images.h>

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace naivebayes {

class BasicTrainingModel {
 public:
  size_t num_classes_;
  size_t image_size_;
  std::vector<size_t> classes_;

  /**
   * Overloads the >> operator to populate class_probabilities_ and
   * pixel_probabilities_ with data stored in file.
   * @param is: The input stream reference that calls the operator.
   * @param model: The instance of BasicTrainingModel being modified by the
   * operator.
   * @return the istream reference.
   */
  friend std::istream& operator>>(std::istream& is, BasicTrainingModel& model);

  /**
   * Overloads the << operator to write the necessary data into a file.
   * @param os: The output stream reference that calls the operator.
   * @param model: The instance of BasicTrainingModel writing data.
   * @return the ostream reference.
   */
  friend std::ostream& operator<<(std::ostream& os,
                                  const BasicTrainingModel& model);

  /**
   * Reads the file at the passed file path and stores the labels in a vector.
   * @param file_path
   */
  void ReadLabels(const std::string& file_path);

  /**
   * Public helper function so that user can train the model_.
   * Calls the calculate functions.
   */
  void TrainModel();

  double GetPixelProbability(const size_t class_number, const size_t shade,
                             const size_t row, const size_t col) const;

  double GetClassProbability(const size_t class_number) const;

  /**
   * Gets all class probabilities, then all pixel probabilities.
   * @return a vector of doubles with the above probabilities.
   */
  std::vector<double> GetProbabilities() const;

  std::vector<size_t> GetLabels() const;

  void SetImages(const Images& data_to_add);

 private:
  std::vector<size_t> image_labels_;
  std::unordered_map<size_t, size_t> class_sizes_;
  std::unordered_map<size_t, double> class_probabilities_;
  Images training_images_;

  // An unordered map of ints (class_number) to 2d vectors of doubles
  // (probabilities).
  std::unordered_map<size_t, std::vector<std::vector<double>>>
      pixel_probabilities_;

  // Smoothing value for naive bayes.
  constexpr static const double kLaplaceSmoothingValue = 1;
  static const size_t kShaded = 1;

  /**
   * Uses the formula below to calculate the class probability for each distinct
   * class in the training data. Then stores each probability into an unordered
   * map with key: class (size_t) and value: probability (double).
   *
   * P(class = c) = (k + # of images belonging to class c) / (10k +
   * Total # of training images)
   */
  void CalculateClassProbability();

  /**
   * Uses the formula below to calculate the pixel probability for each pixel,
   * shade, and class. Writes each probability into an unordered map with the
   * key being the class.
   *
   * P(Fi,j = f | class = c) = (k + # of images belonging to class c
   * where Fi,j = f) / (2k + Total # of images belonging to class c)
   */
  void CalculatePixelProbability();

  /**
   * Helper function that counts the number of classes and the size of each class.
   * Stores the number of classes in num_classes_ and each class in classes_.
   */
  void CountSizeNumClasses();
};

}  // namespace naivebayes

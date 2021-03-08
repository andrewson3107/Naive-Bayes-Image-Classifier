#include <core/basic_training_model.h>
#include <core/classifier.h>
#include <gflags/gflags.h>

#include <fstream>
#include <iostream>

DEFINE_string(read_images, "", "Specify a file path for the training images");
DEFINE_string(read_labels, "", "Specify a file path for the training labels");
DEFINE_string(save, "", "Specify a file path to load probability data to");
DEFINE_string(load, "", "Specify a file path to load probability data from");
DEFINE_string(read_test_images, "",
              "Specify a file path for the testing images");
DEFINE_string(read_test_labels, "",
              "Specify a file path for the testing labels");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  naivebayes::BasicTrainingModel model;
  naivebayes::Images data;
  naivebayes::Classifier classifier;

  if (FLAGS_read_images.empty() && FLAGS_read_labels.empty() &&
      FLAGS_save.empty() && FLAGS_load.empty()) {
    std::cout
        << "No arguments were provided, or provided arguments are incorrect."
        << std::endl;
  }

  if (!FLAGS_read_images.empty() && !FLAGS_read_labels.empty()) {
    std::ifstream ifs(FLAGS_read_images);
    ifs >> data;
    model.SetImages(data);
    std::cout << "Images successfully read." << std::endl;
    model.ReadLabels(FLAGS_read_labels);
    std::cout << "Labels successfully read." << std::endl;
    model.TrainModel();

    if (!FLAGS_save.empty()) {
      std::ofstream ofs(FLAGS_save);
      ofs << model;
      std::cout << "Data successfully saved." << std::endl;
    }

  } else {
    if (!FLAGS_save.empty()) {
      std::ofstream ofs(FLAGS_save);
      ofs << model;
      std::cout << "Model could not be saved because it was not trained."
                << std::endl;
    }
  }

  if (!FLAGS_load.empty()) {
    std::ifstream ifs(FLAGS_load);
    ifs >> classifier.model_;
    std::cout << "Data successfully loaded into model." << std::endl;
  }

  if (!FLAGS_read_test_images.empty() && !FLAGS_read_test_labels.empty()) {
    std::ifstream ifs(FLAGS_read_test_images);
    ifs >> data;
    classifier.ReadLabels(FLAGS_read_test_labels);

    std::cout << "Accuracy: " << classifier.CalculateAccuracy(data)
              << std::endl;
  } else if (!FLAGS_read_test_images.empty() ||
             !FLAGS_read_test_labels.empty()) {
    std::cout << "A test file was missing." << std::endl;
  }

  return 0;
}

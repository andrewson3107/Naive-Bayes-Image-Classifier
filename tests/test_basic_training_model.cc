#include <core/basic_training_model.h>
#include <catch2/catch.hpp>
#include <fstream>


using naivebayes::BasicTrainingModel;
using naivebayes::Image;
using naivebayes::Images;
using std::string;

std::string test_images_file_path =
    "c:/Users/Andori/Cinder/my-projects/naivebayes-andrewson3107/tests/data/"
    "testing_training_images.txt";
std::string test_labels_file_path =
    "c:/Users/Andori/Cinder/my-projects/naivebayes-andrewson3107/tests/data/"
    "testing_training_labels.txt";
std::string probability_data_file_path =
    "c:/Users/Andori/Cinder/my-projects/naivebayes-andrewson3107/tests/data/"
    "testing_data.txt";

TEST_CASE("Saving and loading model") {
  SECTION("File does not exist") {
    BasicTrainingModel model;
    Images test_data;
    SECTION("Image file") {
      std::ifstream ifs("fes/dw.txt");
      REQUIRE_THROWS_AS(ifs >> test_data, std::invalid_argument);
    }
    SECTION("Label file") {
      REQUIRE_THROWS_AS(model.ReadLabels("IOD/dsad/d"), std::invalid_argument);
    }
    SECTION("Data file") {
      std::ifstream ifs("fes/dw.txt");
      REQUIRE_THROWS_AS(ifs >> model, std::invalid_argument);
    }
  }

  SECTION("Data is correct") {
    // Reads the test images_, creates a model_, trains the model_, then writes
    // the data to a file.
    BasicTrainingModel model1;
    Images test_data;

    std::ifstream ifs1(test_images_file_path);
    ifs1 >> test_data;

    model1.ReadLabels(test_labels_file_path);
    model1.SetImages(test_data);
    model1.TrainModel();
    std::ofstream ofs(probability_data_file_path);
    ofs << model1;

    // Creates another model_ but does not read the test images_. Instead, loads
    // data from file.
    BasicTrainingModel model2;
    std::ifstream ifs2(probability_data_file_path);
    ifs2 >> model2;

    SECTION("Number of classes") {
      REQUIRE(model2.num_classes_ == model1.num_classes_);
    }

    SECTION("Probabilities") {
      std::vector<double> model1_probabilities = model1.GetProbabilities();
      std::vector<double> model2_probabilities = model2.GetProbabilities();
      for (size_t index = 0; index < model1_probabilities.size(); index++) {
        REQUIRE(model1_probabilities[index] ==
                Approx(model2_probabilities[index]));
      }
    }

  }
}

TEST_CASE("Calculating probabilities") {
  BasicTrainingModel model;
  Images test_data;
  std::ifstream ifs(test_images_file_path);
  ifs >> test_data;
  model.SetImages(test_data);
  model.ReadLabels(test_labels_file_path);
  model.TrainModel();
  std::ofstream ofs(probability_data_file_path);
  ofs << model;

  std::vector<double> expected_values = {
      0.4,      0.6,      0.333333, 0.333333, 0.333333, 0.333333, 0.666667,
      0.333333, 0.333333, 0.333333, 0.333333, 0.5,      0.25,     0.75,
      0.75,     0.25,     0.75,     0.5,      0.25,     0.5};

  std::vector<double> actual_values = model.GetProbabilities();

  for (size_t index = 0; index < expected_values.size(); index++) {
    REQUIRE(expected_values[index] == Approx(actual_values[index]));
  }
}

TEST_CASE("Reading in files") {
  Image image1 = {{'#', '#', '#'}, {'#', ' ', '#'}, {'#', '#', '#'}};
  Image image2 = {{'#', '#', ' '}, {' ', '#', ' '}, {'#', '#', '#'}};
  Image image3 = {{' ', '#', ' '}, {' ', '#', ' '}, {' ', '#', ' '}};

  BasicTrainingModel model;
  Images test_data;
  std::ifstream ifs(test_images_file_path);
  ifs >> test_data;
  model.SetImages(test_data);
  model.ReadLabels(test_labels_file_path);

  SECTION("Labels are correct") {
    std::vector<size_t> expected_labels = {0, 1, 1};
    std::vector<size_t> actual_labels = model.GetLabels();
    for (size_t index = 0; index < model.num_classes_; index++) {
      REQUIRE(expected_labels[index] == actual_labels[index]);
    }
  }

  SECTION("Images are correct") {
    SECTION("Image 1") {
      for (size_t row = 0; row < image1.size(); row++) {
        for (size_t col = 0; col < image1.size(); col++) {
          REQUIRE(image1[row][col] == test_data.GetPixel(0, row, col));
        }
      }
    }

    SECTION("Image 2") {
      for (size_t row = 0; row < image1.size(); row++) {
        for (size_t col = 0; col < image1.size(); col++) {
          REQUIRE(image2[row][col] == test_data.GetPixel(1, row, col));
        }
      }
    }

    SECTION("Image 3") {
      for (size_t row = 0; row < image1.size(); row++) {
        for (size_t col = 0; col < image1.size(); col++) {
          REQUIRE(image3[row][col] == test_data.GetPixel(2, row, col));
        }
      }
    }
  }
}

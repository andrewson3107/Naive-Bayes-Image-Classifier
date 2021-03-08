#include <core/classifier.h>

#include <catch2/catch.hpp>
#include <fstream>

using naivebayes::BasicTrainingModel;
using naivebayes::Classifier;
using naivebayes::Image;
using naivebayes::Images;

TEST_CASE("Mathematical correctness") {
  Classifier classifier;
  std::ifstream ifs(
      "c:/Users/Andori/Cinder/my-projects/naivebayes-andrewson3107/tests/data/"
      "testing_data.txt");
  ifs >> classifier.model_;

  Images test_images;
  std::ifstream ifs2(
      "c:/Users/Andori/Cinder/my-projects/naivebayes-andrewson3107/tests/"
      "data/"
      "testing_test_images.txt");
  ifs2 >> test_images;
  std::vector<Image> images_to_classify = test_images.GetImages();

  SECTION("Likelihood calculations are correct") {
    std::vector<double> expected_scores = {-3.788943, -1.874583, -2.584820,
                                           -3.783056, -2.885851, -3.783056};

    SECTION("Image 1") {
      REQUIRE(classifier.CalculateLikelihoodScore(0, images_to_classify[0]) ==
              Approx(expected_scores[0]));
      REQUIRE(classifier.CalculateLikelihoodScore(1, images_to_classify[0]) ==
              Approx(expected_scores[1]));
    }

    SECTION("Image 2") {
      REQUIRE(classifier.CalculateLikelihoodScore(0, images_to_classify[1]) ==
              Approx(expected_scores[2]));
      REQUIRE(classifier.CalculateLikelihoodScore(1, images_to_classify[1]) ==
              Approx(expected_scores[3]));
    }

    SECTION("Image 3") {
      REQUIRE(classifier.CalculateLikelihoodScore(0, images_to_classify[2]) ==
              Approx(expected_scores[4]));
      REQUIRE(classifier.CalculateLikelihoodScore(1, images_to_classify[2]) ==
              Approx(expected_scores[5]));
    }
  }

  SECTION("Classifications are correct") {
    std::vector<size_t> expected_class = {1,0,0};

    for (size_t index = 0; index < images_to_classify.size(); index++) {
      REQUIRE(classifier.ClassifyImage(images_to_classify[index]) == expected_class[index]);
    }
  }
}

TEST_CASE("Accuracy is acceptable") {
  Classifier classifier;
  std::ifstream ifs1(
      "c:/Users/Andori/Cinder/my-projects/naivebayes-andrewson3107/data/"
      "savedmodeldata");
  ifs1 >> classifier.model_;

  classifier.ReadLabels(
      "c:/Users/Andori/Cinder/my-projects/naivebayes-andrewson3107/data/"
      "testlabels");

  Images test_images;
  std::ifstream ifs2(
      "c:/Users/Andori/Cinder/my-projects/naivebayes-andrewson3107/data/"
      "testimages");
  ifs2 >> test_images;

  REQUIRE(classifier.CalculateAccuracy(test_images) > .7);
}

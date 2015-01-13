//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>

#define DLL_PARALLEL
#define DLL_SVM_SUPPORT

#include "dll/conv_rbm.hpp"
#include "dll/conv_dbn.hpp"
#include "dll/test.hpp"
#include "dll/ocv_visualizer.hpp"

#include "dll/cpp_utils/algorithm.hpp"
#include "dll/cpp_utils/data.hpp"

#include "icdar/icdar_reader.hpp"

#include <opencv2/opencv.hpp>

static constexpr const std::size_t window = 16;
static constexpr const std::size_t step_size = 4;

static constexpr const std::size_t min_peaks = 2;
static constexpr const std::size_t max_peaks = 8;

static constexpr const std::size_t binary_threshold = 35;

cv::Mat open_image(const std::string& path);

constexpr uint8_t reduce_val(uint8_t val, uint8_t mul){
    //return val;
    //return (val / mul) * mul;
    return val / mul * mul + mul / 2;
    //if (val < 192) return uchar(val / 64.0 + 0.5) * 64;
    //return 255;
    //if (val < 64) return 0;
    //if (val < 128) return 64;
    //return 255;
}

void reduce_image(cv::Mat& image, uint8_t mul){
    for(std::size_t x = 0; x < image.cols; ++x){
        for(std::size_t y = 0; y < image.rows; ++y){
            image.at<cv::Vec3b>(cv::Point(x,y))[0] = reduce_val(image.at<cv::Vec3b>(cv::Point(x,y))[0], mul);
            image.at<cv::Vec3b>(cv::Point(x,y))[1] = reduce_val(image.at<cv::Vec3b>(cv::Point(x,y))[1], mul);
            image.at<cv::Vec3b>(cv::Point(x,y))[2] = reduce_val(image.at<cv::Vec3b>(cv::Point(x,y))[2], mul);
        }
    }
}

template<typename Iterator>
std::size_t count_peaks(Iterator first, Iterator last){
    std::size_t peaks = 0;
    bool current = false;

    while(first != last){
        auto c = *first;

        if(!current && c > 0){
            current = true;
        }

        if(current && c == 0){
            ++peaks;
            current = false;
        }

        ++first;
    }

    return peaks;
}

std::size_t count_peaks(cv::Mat& image, std::size_t channel){
    std::array<uint8_t, 256> intensities;
    std::fill(intensities.begin(), intensities.end(), 0);

    for(auto it = image.begin<cv::Vec3b>(), end = image.end<cv::Vec3b>(); it != end; ++it){
        ++intensities[(*it)[channel]];
    }

    return count_peaks(intensities.begin(), intensities.end());
}

std::size_t count_gray_peaks(cv::Mat& image){
    std::array<uint8_t, 256> intensities;
    std::fill(intensities.begin(), intensities.end(), 0);

    for(auto it = image.begin<uchar>(), end = image.end<uchar>(); it != end; ++it){
        ++intensities[(*it)];
    }

    return count_peaks(intensities.begin(), intensities.end());
}

cv::Mat binarize(cv::Mat& source_image, uint8_t mul){
    auto image = source_image.clone();

    cv::Mat gray_image;
    cv::cvtColor(source_image, gray_image, CV_RGB2GRAY);

    //cv::namedWindow("Gray", cv::WINDOW_AUTOSIZE);
    //cv::imshow("Gray", gray_image);

    cv::Mat binary_map_image(image.rows, image.cols, CV_8U);
    binary_map_image = cv::Scalar(0);

    std::vector<std::size_t> intensity_map(image.cols * image.rows, 0);

    reduce_image(image, mul);
    //reduce_image(gray_image, mul);

    //cv::namedWindow("Reduced", cv::WINDOW_AUTOSIZE);
    //cv::imshow("Reduced", image);

    //cv::namedWindow("Reduced Gray", cv::WINDOW_AUTOSIZE);
    //cv::imshow("Reduced Gray", gray_image);

    for(std::size_t x = 0; x * step_size + window < image.cols; ++x){
        for(std::size_t y = 0; y * step_size + window < image.rows; ++y){
            cv::Rect rect(x*step_size, y*step_size, window,window);
            cv::Mat roi(image, rect);
            cv::Mat gray_roi(gray_image, rect);

            auto r_peaks = count_peaks(roi, 0);
            auto g_peaks = count_peaks(roi, 1);
            auto b_peaks = count_peaks(roi, 2);
            auto gray_peaks = count_gray_peaks(gray_roi);

            //auto value = gray_peaks > 1 [>&& gray_peaks < max_peaks <] ? 1 : 0;

            std::size_t value =
                    (r_peaks > min_peaks && r_peaks < max_peaks ? 1 : 0)
                +   (g_peaks > min_peaks && g_peaks < max_peaks ? 1 : 0)
                +   (b_peaks > min_peaks && b_peaks < max_peaks ? 1 : 0);

            if(value > 0){
                for(std::size_t xx = 0; xx < window; ++xx){
                    for(std::size_t yy = 0; yy < window; ++yy){
                        intensity_map[(xx + x * step_size) * image.rows + yy + y * step_size] += value;
                    }
                }
            }
        }
    }

    for(std::size_t x = 0; x < image.cols; ++x){
        for(std::size_t y = 0; y < image.rows; ++y){
            if(x > image.cols - window){
                if(intensity_map[x * image.rows + y] < (binary_threshold - 5 * (image.cols - x))){
                    binary_map_image.at<uchar>(cv::Point(x,y)) = 0;
                } else {
                    binary_map_image.at<uchar>(cv::Point(x,y)) = 255;
                }
            } else {
                if(intensity_map[x * image.rows + y] < binary_threshold){
                    binary_map_image.at<uchar>(cv::Point(x,y)) = 0;
                } else {
                    binary_map_image.at<uchar>(cv::Point(x,y)) = 255;
                }
            }
        }
    }

    return binary_map_image;
}

cv::Mat combine(const std::vector<cv::Mat>& binary_maps){
    if(binary_maps.size() == 1){
        return binary_maps[0];
    }

    cv::Mat binary_map_image(binary_maps[0].rows, binary_maps[0].cols, CV_8U);
    binary_map_image = cv::Scalar(0);

    for(auto& binary_map : binary_maps){
        for(std::size_t x = 0; x < binary_map.cols; ++x){
            for(std::size_t y = 0; y < binary_map.rows; ++y){
                if(binary_map.at<uchar>(cv::Point(x,y)) == 255){
                    binary_map_image.at<uchar>(cv::Point(x,y)) = 255;
                }
            }
        }
    }

    return binary_map_image;
}

void process_image(const std::string& source_path, bool display = false){
    auto image = open_image(source_path);

    std::vector<cv::Mat> binary_maps;

    binary_maps.push_back(binarize(image, 32));
    binary_maps.push_back(binarize(image, 48));
    binary_maps.push_back(binarize(image, 64));

    auto binary_map_image = combine(binary_maps);

    auto structure_elem = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(9, 9));
    cv::morphologyEx(binary_map_image, binary_map_image, cv::MORPH_OPEN, structure_elem);
    cv::morphologyEx(binary_map_image, binary_map_image, cv::MORPH_CLOSE, structure_elem);

    auto dst_image = image.clone();
    for(std::size_t x = 0; x < image.cols; ++x){
        for(std::size_t y = 0; y < image.rows; ++y){
            if(binary_map_image.at<uchar>(cv::Point(x,y)) != 255){
                dst_image.at<cv::Vec3b>(cv::Point(x,y))[0] = 0;
                dst_image.at<cv::Vec3b>(cv::Point(x,y))[1] = 0;
                dst_image.at<cv::Vec3b>(cv::Point(x,y))[2] = 0;
            }
        }
    }

    auto dest_path = source_path;
    dest_path.insert(dest_path.rfind('.'), ".map");
    imwrite(dest_path.c_str(), dst_image);

    if(display){
        cv::namedWindow("Source", cv::WINDOW_AUTOSIZE);
        cv::imshow("Source", image);

        cv::namedWindow("Dest", cv::WINDOW_AUTOSIZE);
        cv::imshow("Dest", dst_image);

        cv::namedWindow("Map", cv::WINDOW_AUTOSIZE);
        cv::imshow("Map", binary_map_image);

        cv::waitKey(0);
    }
}

int main(int argc, char* argv[]){
    if(argc > 1){
        std::string source_path = "/home/wichtounet/datasets/icdar_2013_natural_wip/train/";
        source_path += argv[1];
        source_path += ".jpg";
        process_image(source_path, true);
    } else {
        for(std::size_t i = 100; i <= 328; ++i){
            std::string source_path = "/home/wichtounet/datasets/icdar_2013_natural_wip/train/" + std::to_string(i) + ".jpg";
            process_image(source_path);
        }
    }

    return 0;
}

cv::Mat open_image(const std::string& path){
    auto source_image = cv::imread(path.c_str(), 1);

    if (!source_image.data){
        return source_image;
    }

    if(source_image.rows > 1000 || source_image.cols > 1000){
        auto factor = 1000.0f / std::max(source_image.rows, source_image.cols);

        cv::Mat resized_image;

        cv::resize(source_image, resized_image, cv::Size(), factor, factor, cv::INTER_AREA);

        return resized_image;
    }

    return source_image;
}

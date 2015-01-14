//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
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
    return val / mul * mul + mul / 2;
}

void reduce_image(cv::Mat& image, uint8_t mul){
    for(std::size_t x = 0; x < static_cast<std::size_t>(image.cols); ++x){
        for(std::size_t y = 0; y < static_cast<std::size_t>(image.rows); ++y){
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

    //static std::size_t zzz = 0;

    //if(zzz == 666[> || count_gray_peaks(intensities.begin(), intensities.end()) > 1<]){
        //for(std::size_t c = 0; c < 256; ++c){
            //std::cout << c << ":" << static_cast<std::size_t>(intensities[c]) << std::endl;
        //}

        //std::cout << "peaks => " << count_peaks(intensities.begin(), intensities.end()) << std::endl;
    //}

    //++zzz;

    return count_peaks(intensities.begin(), intensities.end());
}

template<typename Iterator>
std::size_t count_gray_peaks(Iterator first, Iterator last){
    std::size_t peaks = 0;

    bool current = false;
    bool up = false;
    std::size_t distance = 0;
    std::size_t prev = 0;
    std::size_t intensity = 0;

    while(first != last){
        auto c = *first;

        if(current){

            if(c < 1){
                ++distance;

                if(distance > 5){
                    current = false;

                    if(intensity > 25){
                        ++peaks;
                    }
                }
            } else {
                distance = 0;
                prev = c;
                if(c > intensity){
                    intensity = c;
                }
            }


        } else {


            if(c > 1){
                current = true;
                up = true;
                prev = c;
                distance = 0;
                intensity = c;
            }



        }

        ++first;
    }

    if(current && distance <= 5 && intensity > 25){
        ++peaks;
    }


    return peaks;
}

std::size_t count_gray_peaks(cv::Mat& image){
    std::array<uint8_t, 256> intensities;
    std::fill(intensities.begin(), intensities.end(), 0);

    for(auto it = image.begin<uchar>(), end = image.end<uchar>(); it != end; ++it){
        ++intensities[(*it)];
    }

    //static std::size_t zzz = 0;

    //if(zzz == 3333 || count_gray_peaks(intensities.begin(), intensities.end()) > 1){
        //for(std::size_t c = 0; c < 256; ++c){
            //std::cout << c << ":" << static_cast<std::size_t>(intensities[c]) << std::endl;
        //}

        //std::cout << "peaks => " << count_gray_peaks(intensities.begin(), intensities.end()) << std::endl;
    //}

    //++zzz;

    return count_gray_peaks(intensities.begin(), intensities.end());
}

void quantize_image(cv::Mat& image, cv::Mat& quant_image){
    std::size_t width = image.cols;
    std::size_t height = image.rows;

    for(std::size_t x = 0; x < width; ++x){
        for(std::size_t y = 0; y < height; ++y){
            uchar c1 = (image.at<cv::Vec3b>(cv::Point(x, y))[0] & 192) >> 2;
            uchar c2 = (image.at<cv::Vec3b>(cv::Point(x, y))[1] & 192) >> 4;
            uchar c3 = (image.at<cv::Vec3b>(cv::Point(x, y))[2] & 192) >> 6;

            quant_image.at<uchar>(cv::Point(x, y)) = c1 | c2 | c3;
        }
    }
}

cv::Mat binarize(cv::Mat& source_image){
    auto image = source_image.clone();

    std::size_t width = image.cols;
    std::size_t height = image.rows;

    cv::Mat binary_map_image(image.rows, image.cols, CV_8U);
    binary_map_image = cv::Scalar(0);

    cv::Mat quant_image(image.rows, image.cols, CV_8U);
    quant_image = cv::Scalar(0);

    quantize_image(image, quant_image);

    std::vector<std::size_t> intensity_map(image.cols * image.rows, 0);

    for(std::size_t x = 0; x * step_size + window < width; ++x){
        for(std::size_t y = 0; y * step_size + window < height; ++y){
            cv::Rect rect(x*step_size, y*step_size, window,window);
            cv::Mat roi(quant_image, rect);

            auto peaks = count_peaks(roi, 0);

            std::size_t value = peaks > min_peaks && peaks < max_peaks ? 3 : 0;

            if(value > 0){
                for(std::size_t xx = 0; xx < window; ++xx){
                    for(std::size_t yy = 0; yy < window; ++yy){
                        intensity_map[(xx + x * step_size) * height + yy + y * step_size] += value;
                    }
                }
            }
        }
    }

    for(std::size_t x = 0; x < width; ++x){
        for(std::size_t y = 0; y < height; ++y){
            if(x > image.cols - window){
                if(intensity_map[x * height + y] < (binary_threshold - 5 * (width - x))){
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

    std::size_t width = binary_maps[0].cols;
    std::size_t height = binary_maps[0].rows;

    cv::Mat binary_map_image(height, width, CV_8U);
    binary_map_image = cv::Scalar(0);

    for(auto& binary_map : binary_maps){
        for(std::size_t x = 0; x < width; ++x){
            for(std::size_t y = 0; y < height; ++y){
                if(binary_map.at<uchar>(cv::Point(x,y)) == 255){
                    binary_map_image.at<uchar>(cv::Point(x,y)) = 255;
                }
            }
        }
    }

    return binary_map_image;
}

void process_image(const std::string& source_path, bool bw = true, bool display = false){
    std::cout << "Process image " << source_path << std::endl;
    auto image = open_image(source_path);
    auto dst_image = image.clone();

    std::vector<cv::Mat> binary_maps;
    binary_maps.push_back(binarize(image));

    auto binary_map_image = combine(binary_maps);

    auto structure_elem = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
    cv::morphologyEx(binary_map_image, binary_map_image, cv::MORPH_OPEN, structure_elem);
    cv::morphologyEx(binary_map_image, binary_map_image, cv::MORPH_CLOSE, structure_elem);

    if(bw){
        for(std::size_t x = 0; x < static_cast<std::size_t>(image.cols); ++x){
            for(std::size_t y = 0; y < static_cast<std::size_t>(image.rows); ++y){
                if(binary_map_image.at<uchar>(cv::Point(x,y)) != 255){
                    dst_image.at<cv::Vec3b>(cv::Point(x,y))[0] = 0;
                    dst_image.at<cv::Vec3b>(cv::Point(x,y))[1] = 0;
                    dst_image.at<cv::Vec3b>(cv::Point(x,y))[2] = 0;
                }
            }
        }

        cv::Mat lines_image;
        constexpr const size_t CANNY_THRESHOLD = 60;
        cv::Canny(binary_map_image, lines_image, CANNY_THRESHOLD, CANNY_THRESHOLD * 3, 5);

        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(lines_image, lines, 1, CV_PI/180, 32, 10, 5);

        //Enlarge a bit the lines
        for(auto& l : lines){
            cv::Vec2f u(l[2] - l[0], l[3] - l[1]);
            u *= 0.02;

            l[2] += u[0];
            l[3] += u[1];

            l[0] -= u[0];
            l[1] -= u[1];
        }

        for(auto& l : lines){
            cv::line(dst_image, cv::Point2f(l[0], l[1]), cv::Point2f(l[2], l[3]), cv::Scalar(0, 255, 255), 2, CV_AA);
        }

    } else {
        cv::Mat canny_image;
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;

        cv::Canny(binary_map_image, canny_image, 100, 100*2, 3);

        cv::findContours(canny_image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

        cv::RNG rng(12345);

        std::size_t c = 0;

        for(int i = 0; i< contours.size(); i++){
            //Ignore the global contour
            if(hierarchy[i][3] < 0){
                continue;
            }

            auto rect_roi = cv::boundingRect(contours[i]);
            cv::Mat roi(binary_map_image, rect_roi);

            auto mean_color = cv::mean(roi)[0];

            //This is mostly black
            if(mean_color < 100){
                continue;
            }

            auto area = cv::contourArea(contours[i]);
            std::cout << "contour " << i << " area = " << area << std::endl;
            std::cout << "contour " << i << " rect area = " << rect_roi.area() << std::endl;
            std::cout << "contour " << i << " area ratio = " << static_cast<double>(area) / rect_roi.area() << std::endl;

            cv::Scalar color(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            //cv::drawContours(dst_image, contours, i, color, 2, 8, hierarchy, 0, cv::Point() );

            cv::rectangle(dst_image, rect_roi, color, 2, 8, 0);

            ++c;
        }

        std::cout << "   " << c << " contours found" << std::endl;
    }

    auto dest_path = source_path;
    dest_path.insert(dest_path.rfind('.'), bw ? ".zzz.map" : ".zzz.contours");
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

        if(argc > 2){
            std::string arg(argv[2]);
            if(arg == "bw"){
                process_image(source_path, true, true);
            } else if(arg == "contours" || arg == "ct"){
                process_image(source_path, false, true);
            } else {
                printf("Unknown mode \"%s\"\n", arg.c_str());
            }
        } else {
            process_image(source_path, false, true);
        }
    } else {
        for(std::size_t i = 100; i <= 328; ++i){
            std::string source_path = "/home/wichtounet/datasets/icdar_2013_natural_wip/train/" + std::to_string(i) + ".jpg";
            process_image(source_path, true);
            process_image(source_path, false);
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

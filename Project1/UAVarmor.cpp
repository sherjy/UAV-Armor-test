
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <iostream>
#include <math.h>
#include <algorithm>

using namespace std;
using namespace cv;

#define TRACKING_MAX_MATCH_THRE 0
#define TRACKING_FRAME_MAX_NUM 8

//send the x and y coordinate to base
typedef union
{
	int DATA[2];
	struct {
		int xlocation;
		int ylocation;
	}location;
}armour_send_location;


//find the rotated rectangle of a light on a armour
vector<RotatedRect> get_fit_rRect(Mat mask, Mat img_draw) {
	vector<RotatedRect> rRect_list;
	vector<vector<Point> > contours;
	float aspect_ratio;
	float rRect_height;
	float rRect_angle;
	RotatedRect output_rRect;
	findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
	for (int i = 0; i < contours.size(); i++) {
		if (contours[i].size()>10) {
			RotatedRect rRect_fit = minAreaRect(contours[i]);
			RotatedRect Elli_fit = fitEllipse(contours[i]);
			if (rRect_fit.size.height>rRect_fit.size.width) {
				aspect_ratio = rRect_fit.size.height / rRect_fit.size.width;
				rRect_height = rRect_fit.size.height;
				rRect_angle = Elli_fit.angle;
			}
			else {
				aspect_ratio = rRect_fit.size.width / rRect_fit.size.height;
				rRect_height = rRect_fit.size.width;
				rRect_angle = Elli_fit.angle;
			}
			if (rRect_angle > 90) {
				rRect_angle -= 180;
			}
			if (abs(rRect_angle)<20 && rRect_height>11 && aspect_ratio>1.5) {
				//char output_txt_a[256];
				//char output_txt_h[256];
				//char output_txt_ar[256];
				//sprintf(output_txt_a, "A:%4.2f", rRect_angle);
				//sprintf(output_txt_h, "H:%4.2f", rRect_height);
				//sprintf(output_txt_ar, "AR:%4.2f", aspect_ratio);
				//putText(img_draw, output_txt_a, Point(rRect_fit.center.x, rRect_fit.center.y + 10), CV_FONT_HERSHEY_COMPLEX, 0.3, Scalar(255, 0, 0));
				//putText(img_draw, output_txt_h, Point(rRect_fit.center.x, rRect_fit.center.y + 20), CV_FONT_HERSHEY_COMPLEX, 0.3, Scalar(255, 0, 0));
				//putText(img_draw, output_txt_ar, Point(rRect_fit.center.x, rRect_fit.center.y + 30), CV_FONT_HERSHEY_COMPLEX, 0.3, Scalar(255, 0, 0));
				output_rRect.angle = Elli_fit.angle;
				output_rRect.center = Elli_fit.center;
				output_rRect.size.height = max(rRect_fit.size.height, rRect_fit.size.width);
				output_rRect.size.width = min(rRect_fit.size.height, rRect_fit.size.width);
				rRect_list.push_back(output_rRect);
				circle(img_draw, rRect_fit.center, 4, Scalar(255, 0, 0), -1);
			}
		}
	}
	//imshow("find the light", img_draw);
	return rRect_list;
}


//determine if two rRect are from one armour
int get_match_result(RotatedRect rRect_1, RotatedRect rRect_2) {
	double aspect_ratio_1;
	double rRect_height_1;
	double rRect_angle_1;
	double aspect_ratio_2;
	double rRect_height_2;
	double rRect_angle_2;
	double distance_height_ratio;
	double height_bias_ratio;
	double y_bias_ratio;
	double angle_bias;
	rRect_height_1 = rRect_1.size.height;
	rRect_angle_1 = rRect_1.angle;
	if (rRect_angle_1 > 90) {
		rRect_angle_1 -= 180;
	}
	rRect_height_2 = rRect_2.size.height;
	rRect_angle_2 = rRect_2.angle;
	if (rRect_angle_2 > 90) {
		rRect_angle_2 -= 180;
	}
	distance_height_ratio = fabs(rRect_1.center.x - rRect_2.center.x) / (max(rRect_height_1, rRect_height_2));
	height_bias_ratio = fabs(rRect_height_1 - rRect_height_2) / (max(rRect_height_1, rRect_height_2));
	y_bias_ratio = fabs(rRect_1.center.y - rRect_2.center.y) / (max(rRect_height_1, rRect_height_2));
	angle_bias = fabs((rRect_angle_1 - rRect_angle_2));
	cout << distance_height_ratio << " " << height_bias_ratio << " " << angle_bias << " " << y_bias_ratio << "###########" << endl;
	if (distance_height_ratio>1.7 && distance_height_ratio<4 && height_bias_ratio<0.08 && angle_bias<10 && y_bias_ratio<1.2) {
		cout << distance_height_ratio << " " << height_bias_ratio << " " << angle_bias << " " << y_bias_ratio << "*********" << endl;
		return 1;
	}
	else {
		return 0;
	}
}

float get_distance(Point2i center_1, Point2i center_2) {
	return (center_1.x - center_2.x)*(center_1.x - center_2.x) + (center_1.y - center_2.y)*(center_1.y - center_2.y);
}

//match the selected rotated rectange of lights. Two rotated rectangle will be matched if they are from one armour.
Rect match_rRect(vector<RotatedRect> rRect_list, Mat img_draw, Rect prior_box, Mat mask) {
	vector<Rect> matched_Rect_list;
	vector<float> distance_list;
	Point prior_center;
	prior_center.x = prior_box.x + int(prior_box.width / 2);
	prior_center.y = prior_box.y + int(prior_box.height / 2);
	Rect no_rect(0, 0, 0, 0);
	if (rRect_list.size() > 0) {
		for (int i = 0; i < rRect_list.size() - 1; i++) {
			for (int j = i + 1; j < rRect_list.size(); j++) {
				if (get_match_result(rRect_list[i], rRect_list[j]) == 1) {
					Rect armour;
					Rect light1;
					Rect light2;
					Point2i armour_center;
					Rect in_rect;
					Mat in_mat;
					light1 = rRect_list[i].boundingRect();
					light2 = rRect_list[j].boundingRect();
					armour.x = max(1, min(light1.x, light2.x));
					armour.y = max(1, min(light1.y, light2.y));
					armour.height = min((img_draw.rows - armour.y - 2), (max(light1.br().y, light2.br().y) - armour.y - 1));
					armour.width = min((img_draw.cols - armour.x - 2), (max(light1.br().x, light2.br().x) - armour.x - 1));
					in_rect.x = armour.x + max(light1.width, light2.width);
					in_rect.y = armour.y;
					in_rect.width = armour.width - 2 * max(light1.width, light2.width);
					in_rect.height = armour.height;
					if (in_rect.width > 0 && in_rect.height > 0) {
						mask(in_rect).copyTo(in_mat);
						vector<vector<Point> > in_contours;
						findContours(in_mat, in_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
						//imshow("in_mat", in_mat);
						if (in_contours.size() == 0) {
							matched_Rect_list.push_back(armour);
							armour_center.x = int(armour.x + armour.width / 2);
							armour_center.y = int(armour.y + armour.height / 2);
							distance_list.push_back(get_distance(prior_center, armour_center));
						}
					}
				}
			}
		}
		if (matched_Rect_list.size() > 0) {
			vector<float>::iterator min_distance = min_element(begin(distance_list), end(distance_list));
			int min_position = distance(std::begin(distance_list), min_distance);
			return matched_Rect_list[min_position];
		}
		else {
			return no_rect;
		}
	}
	else {
		return no_rect;
	}
}

Rect tracking(Mat img, Rect prior_rect, Mat patch) {
	Rect search_ROI;
	Mat search_area;
	Mat match_result;
	Point match_point;
	double max_match_value;
	Rect tracking_match_rect(0, 0, 0, 0);
	search_ROI.x = max(1, prior_rect.x - 1 * prior_rect.width);
	search_ROI.y = max(1, prior_rect.y - 1 * prior_rect.height);
	search_ROI.width = min((img.cols - search_ROI.x - 2), 3 * prior_rect.width);
	search_ROI.height = min((img.rows - search_ROI.y - 2), 3 * prior_rect.height);
	//cout << search_ROI.x << " " << search_ROI.y << " " << search_ROI.width << " " << search_ROI.height << endl;
	img(search_ROI).copyTo(search_area);
	//imshow("patch", patch);
	//imshow("search_area", search_area);
	matchTemplate(search_area, patch, match_result, CV_TM_CCOEFF_NORMED);
	minMaxLoc(match_result, NULL, &max_match_value, NULL, &match_point);
	//imshow("match_result", match_result);
	//cout << "match_point   " << match_point << endl;
	if (max_match_value > TRACKING_MAX_MATCH_THRE) {
		tracking_match_rect.x = max(1, match_point.x + search_ROI.x);
		tracking_match_rect.y = max(1, match_point.y + search_ROI.y);
		tracking_match_rect.width = min((img.cols - tracking_match_rect.x - 2), prior_rect.width);
		tracking_match_rect.height = min((img.rows - tracking_match_rect.y - 2), prior_rect.height);

	}
	else {
		tracking_match_rect.x = 0;
		tracking_match_rect.y = 0;
		tracking_match_rect.width = 0;
		tracking_match_rect.height = 0;
	}
	return tracking_match_rect;
}

Mat get_mask(Mat img_gray) {
	Mat mask;
	threshold(img_gray, mask, 150, 255, CV_THRESH_BINARY);
	Mat element_dila_first = getStructuringElement(MORPH_RECT, Size(1, 3));
	Mat element_open = getStructuringElement(MORPH_RECT, Size(2, 2));
	Mat element_close = getStructuringElement(MORPH_RECT, Size(3, 1));
	morphologyEx(mask, mask, MORPH_OPEN, element_open);
	morphologyEx(mask, mask, MORPH_DILATE, element_dila_first);
	morphologyEx(mask, mask, MORPH_CLOSE, element_close);
	return mask;
}

int main() {

	string left_video_path = "C:/Users/lenovo/Desktop/ttt/mip.mp4";
	//VideoCapture cap_left(1);
	//if (!(cap_left.isOpened()))
	//return -1;
	VideoCapture cap_left(left_video_path);
	armour_send_location send_data = { 0 };

	Mat left_img_rgb;
	Mat left_img_gray;
	Mat left_binary_mask;
	vector<RotatedRect> light_list;
	Rect left_detect_rect;
	Rect prior_box(0, 0, 0, 0);
	Mat tracking_patch;
	Rect output_rect(0, 0, 0, 0);
	int tracking_frame_num;
	Scalar color;
	Rect right_match_rect;
	Point left_output_point;
	while (1) {
		cap_left >> left_img_rgb;
		cvtColor(left_img_rgb, left_img_gray, CV_BGR2GRAY);
		left_binary_mask = get_mask(left_img_gray);
		imshow("left_binary_mask", left_binary_mask);
		light_list = get_fit_rRect(left_binary_mask, left_img_rgb);
		left_detect_rect = match_rRect(light_list, left_img_rgb, prior_box, left_binary_mask);
		if (prior_box.width == 0 && left_detect_rect.width == 0) {
			output_rect.width = 0;
			tracking_frame_num = 0;
		}
		else if (prior_box.width == 0 && left_detect_rect.width > 0) {
			output_rect = left_detect_rect;
			color = Scalar(0, 255, 0);
			tracking_frame_num = 0;
		}
		else if (prior_box.width > 0 && left_detect_rect.width == 0) {
			if (tracking_frame_num <= TRACKING_FRAME_MAX_NUM) {
				output_rect = tracking(left_img_gray, prior_box, tracking_patch);
				tracking_frame_num += 1;
				color = Scalar(0, 255, 255);
			}
			else {
				output_rect.width = 0;
				tracking_frame_num = 0;
			}
		}
		else {
			if ((abs(prior_box.x - left_detect_rect.x) > 10 || abs(prior_box.y - left_detect_rect.y) > 10) && (left_detect_rect.area()<1.5*prior_box.area()) && (tracking_frame_num <= TRACKING_FRAME_MAX_NUM)) {
				output_rect = tracking(left_img_gray, prior_box, tracking_patch);
				tracking_frame_num += 1;
				color = Scalar(0, 255, 255);
			}
			else {
				output_rect = left_detect_rect;
				color = Scalar(0, 255, 0);
				tracking_frame_num = 0;
			}
		}
		if (output_rect.width > 0) {
			rectangle(left_img_rgb, output_rect, color, 2);
			prior_box = output_rect;
			left_img_gray(output_rect).copyTo(tracking_patch);
			left_output_point.x = int(output_rect.x + output_rect.width / 2);
			left_output_point.y = int(output_rect.y + output_rect.height / 2);
			circle(left_img_rgb, left_output_point, 4, Scalar(255, 0, 0), -1);



			send_data.location.xlocation = left_output_point.x;
			send_data.location.ylocation = left_output_point.y;

			cout << send_data.location.xlocation << " " << send_data.location.ylocation << endl;

		}
		else {
			prior_box.width = 0;
		}
		imshow("left_img_rgb", left_img_rgb);
		waitKey(1);
	}
	waitKey(0);
	return 0;

}



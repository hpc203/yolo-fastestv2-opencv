#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

class yolo_fast
{
public:
	yolo_fast(string modelpath, float objThreshold, float confThreshold, float nmsThreshold);
	void detect(Mat& srcimg);

private:
private:
	const float anchors[2][6] = { {12.64,19.39, 37.88,51.48, 55.71,138.31}, {126.91,78.23, 131.57,214.55, 279.92,258.87} };
	const float stride[3] = { 16.0, 32.0 };
	const int inpWidth = 352;
	const int inpHeight = 352;
	const int num_stage = 2;
	const int anchor_num = 3;
	float objThreshold;
	float confThreshold;
	float nmsThreshold;
	vector<string> classes;
	const string classesFile = "coco.names";
	int num_class;
	Net net;
	void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);
};

yolo_fast::yolo_fast(string modelpath, float obj_Threshold, float conf_Threshold, float nms_Threshold)
{
	this->objThreshold = obj_Threshold;
	this->confThreshold = conf_Threshold;
	this->nmsThreshold = nms_Threshold;

	ifstream ifs(this->classesFile.c_str());
	string line;
	while (getline(ifs, line)) this->classes.push_back(line);
	this->num_class = this->classes.size();
	this->net = readNet(modelpath);
}

void yolo_fast::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)   // Draw the predicted bounding box
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 2);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	label = this->classes[classId] + ":" + label;

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 1);
}

void yolo_fast::detect(Mat& frame)
{
	Mat blob;
	blobFromImage(frame, blob, 1 / 255.0, Size(this->inpWidth, this->inpHeight));
	this->net.setInput(blob);
	vector<Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

	/////generate proposals
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;
	float ratioh = (float)frame.rows / this->inpHeight, ratiow = (float)frame.cols / this->inpWidth;
	int n = 0, q = 0, i = 0, j = 0, nout = this->anchor_num * 5 + this->classes.size(), row_ind = 0;
	float* pdata = (float*)outs[0].data;
	for (n = 0; n < this->num_stage; n++)   ///stage
	{
		int num_grid_x = (int)(this->inpWidth / this->stride[n]);
		int num_grid_y = (int)(this->inpHeight / this->stride[n]);
		for (i = 0; i < num_grid_y; i++)
		{
			for (j = 0; j < num_grid_x; j++)
			{
				Mat scores = outs[0].row(row_ind).colRange(this->anchor_num * 5, outs[0].cols);
				Point classIdPoint;
				double max_class_socre;
				// Get the value and location of the maximum score
				minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
				for (q = 0; q < this->anchor_num; q++)    ///anchor
				{
					const float anchor_w = this->anchors[n][q * 2];
					const float anchor_h = this->anchors[n][q * 2 + 1];
					float box_score = pdata[4 * this->anchor_num + q];
					if (box_score > this->objThreshold && max_class_socre > this->confThreshold)
					{
						float cx = (pdata[4 * q] * 2.f - 0.5f + j) * this->stride[n];  ///cx
						float cy = (pdata[4 * q+ 1] * 2.f - 0.5f + i) * this->stride[n];   ///cy
						float w = powf(pdata[4 * q + 2] * 2.f, 2.f) * anchor_w;   ///w
						float h = powf(pdata[4 * q + 3] * 2.f, 2.f) * anchor_h;  ///h

						int left = (cx - 0.5*w)*ratiow;
						int top = (cy - 0.5*h)*ratioh;   ///���껹ԭ��ԭͼ��

						classIds.push_back(classIdPoint.x);
						confidences.push_back(box_score * max_class_socre);
						boxes.push_back(Rect(left, top, (int)(w*ratiow), (int)(h*ratioh)));
					}
				}
				row_ind++;
				pdata += nout;
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		this->drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
	}
}

int main()
{
	yolo_fast yolo_model("model.onnx", 0.3, 0.3, 0.4);
	string imgpath = "img/000148.jpg";
	Mat srcimg = imread(imgpath);
	yolo_model.detect(srcimg);

	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
}

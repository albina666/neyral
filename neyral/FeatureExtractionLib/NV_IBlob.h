#pragma once
#include "IBlobProcessor.h"
class NV_IBlob :
	public fe::IBlobProcessor
{
public:
	NV_IBlob();
	~NV_IBlob();
	std::string GetType();
	std::vector<cv::Mat> DetectBlobs(cv::Mat image);

	/**
	 * �������� ������ ������� �������� � ������� ��������.
	 * @param blobs - ������� �������, ������ ������������ ����� ����� ������� �� ������ ����, ��� CV_8UC1.
	 * @param normilized_blobs - ����� ��� ������ ������� �������� ������� �������.
	 *               ������� ������� ����� ������ ������������ ����� ����� ����� �� ������ ����, ��� CV_8UC1.
	 * @param side - ������� �������� �� ������� ����� ���������� ��������������� ������� �������.
	 */
	std::vector<cv::Mat> NormalizeBlobs(
		std::vector<cv::Mat>& blobs,
		int side
	);
};


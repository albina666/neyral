#pragma once
#include "PolynomialManager.h"
class NV_PoliMan :
	public fe::PolynomialManager
{
public:
	NV_PoliMan();
	~NV_PoliMan();
	fe::ComplexMoments Decompose(cv::Mat blob);

	/**
	 * ������������ �������� �� ����������.
	 * @param decomposition - ���������� �������� � ���.
	 * @return ��������������� �����������, ����� ��� CV_64FC1.
	 */
	cv::Mat Recovery(fe::ComplexMoments& decomposition);

	/**
	 * ������������������� ����� ������������� ��������� ~ exp(jm*fi).
	 * @param n_max - ������������ ���������� ������� ���������.
	 * @param diameter - ������� ����������, �� ������� ����� ������������� ��������, �������.
	 */
	void InitBasis(int n_max, int diameter);

	/**
	 * �������� �������� ������� ��� ������ � ����������.
	 * @return - �������� �������.
	 */
	std::string GetType();
};


#pragma once
#include "ANN.h"
#include <iostream>

class NeuralRealisation : public ANN::ANeuralNetwork
{
public:
	NeuralRealisation()
	{
	}
	NeuralRealisation(std::vector<size_t> conf, ANeuralNetwork::ActivationType aType, float scale);

	std::string GetType();
	std::vector<float> Predict(std::vector<float> & input);
	
	~NeuralRealisation();
protected:
	void SetConf(std::vector<size_t> conf)
	{
		this->configuration = conf;
	}
	void SetAcT(ANeuralNetwork::ActivationType aType)
	{
		this->activation_type = aType;
	}
	void SetSc(float scale)
	{
		this->scale = scale;
	}
};


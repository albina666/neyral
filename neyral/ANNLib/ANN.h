#pragma once

#ifdef ANNDLL_EXPORTS
#define ANNDLL_API __declspec(dllexport) 
#else
#define ANNDLL_API __declspec(dllimport) 
#endif

#include <vector>
#include <memory>
#include <string>

namespace ANN
{
	class ANeuralNetwork
	{
	public:
		/*
		 * ��������� ���� ������������� �������.
		 */
		enum ActivationType
		{
			POSITIVE_SYGMOID,   // ������������� ����������� ���������.
			BIPOLAR_SYGMOID		// ���������� ��������.
		};

		/**
		* ��������� ��������� ���� �� �����. ���� ����������� ������� ������ Save.
		* @param filepath - ��� � ���� �� ����� � �����.
		* @return - ���������� ����������.
		*/
		ANNDLL_API virtual bool Load(std::string filepath);

		/**
		* ��������� ��������� ���� � ����. ���� ����������� ������� ������ Load.
		* @param filepath - ��� � ���� �� ����� � �����.
		* @return - ���������� ����������.
		*/
		ANNDLL_API virtual bool Save(std::string filepath);

		/**
		* �������� ������������ ����.
		* @return ������������ ���� - ������ - � ������ �������� �������� ���������� �������� � ����.
		*			����� �������� ������������� ������ ����.
		*/
		ANNDLL_API virtual std::vector<size_t> GetConfiguration();

		/**
		* ������������������ ���� ���� ��������� �������.
		*/
		ANNDLL_API void RandomInit();

		/**************************************************************************/
		/**********************��� ��� ���� ����������� �����**********************/
		/**************************************************************************/

		/**
		* �������� ������ � ����� ����.
		* @return �������� ����, �������� ������ � ���� ��������� ���� � ������ ����������.
		*/
		ANNDLL_API virtual std::string GetType() = 0;

		/**
		* ��������������� ����� �� ��������� �����.
		* @param input - ����, ����� ������ ��������������� ���������� �������� �� ������� ����.
		* @return ����� ����, ����� ������������� ���������� �������� � �������� ����.
		*/
		ANNDLL_API virtual std::vector<float> Predict(std::vector<float> & input) = 0;

		/**
		* ������� ��������� ����
		* @param configuration - ������������ ��������� ����.
		*   ������ ������� ������������ ����� ���������� �������� � ��������� ����.
		* @param activation_type - ��� ������������� ������� (�����������, ����������).
		* @param scale - ������� ������������� �������.
		*/
		friend ANNDLL_API std::shared_ptr<ANN::ANeuralNetwork> CreateNeuralNetwork(
			std::vector<size_t> & configuration = std::vector<size_t>(),
			ANeuralNetwork::ActivationType activation_type = ANeuralNetwork::POSITIVE_SYGMOID, 
			float scale = 1.0
		);

		/**
		* ������� ���� ������� ��������� ��������������� ������.
		* � ���� ������ ������, ����� ���������� �������� ���� is_trained ������ ��������������� � true.
		* @param ann - ��������� ����, ������� ���������� �������.
		* @param inputs - ����� ��� ��������.
		* @param outputs - ������ ��� ��������.
		* @param max_iters - ������������ ���������� �������� ��� ��������.
		* @param eps - ������� ������ �� ���� �������� ��� ������� ���������� ��������� ��������.
		* @param speed - �������� ��������.
		* @param std_dump - ���������� �� ���������� � �������� �������� � ����������� ����� ������?
		*/
		friend ANNDLL_API float BackPropTraining(
			std::shared_ptr<ANN::ANeuralNetwork> ann,
			std::vector<std::vector<float>> & inputs,
			std::vector<std::vector<float>> & outputs,
			int maxIters = 10000,
			float eps = 0.1,
			float speed = 0.1,
			bool std_dump = false
		);

		/**
		* �������� ���� �������� �������� ������� ��������� ��������������� ������.
		* @param ann - ��������� ����, ������� ���������� �������.
		* @param input - ���� ��� ��������.
		* @param outputs - ����� ��� ��������.
		* @param speed - �������� ��������.
		*/
		friend ANNDLL_API float BackPropTrainingIteration(
			std::shared_ptr<ANN::ANeuralNetwork> ann, 
			const std::vector<float>& input,
			const std::vector<float>& output,
			float speed
		);

		/***************************************************************************/
		/***************************************************************************/
		/**
		* ����������.
		*/
		ANNDLL_API virtual ~ANeuralNetwork();

	protected:
		/** 
		 * ���� ����. 
		 * ������ ������ - ����� ���� �� �������� ��� �����, 
		 * ������ ������ - ����� ������� �� �������� ��� �����, 
		 * ������ ������ - ����� ������� � �������� ��� �����. 
		 */
		std::vector<std::vector<std::vector<float> > > weights;

		/**
		* ������������ ����.
		* ����� ������� � ������� ������������ ������ ����.
		* �������� - ���������� ��������.
		*/
		std::vector<size_t> configuration;

		/** ������� �� ����? */
		bool is_trained;

		/** �������������� ����������� ��������� ��������. */
		float scale;

		/** ��� ������������� �������. */
		ActivationType activation_type;

		/**
		* ��������� �������� ������������� �������.
		* @param neuron_input - ������� �������� �������.
		* @return - �������� ������������� ������.
		*/
		ANNDLL_API float Activation(float neuron_input);

		/**
		* ��������� �������� ����������� ������������� �������.
		* @param activation - �������� ������������� ��������, ��� ������� ����� ��������� �����������.
		* @return - �������� ����������� ������������� ������.
		*/
		ANNDLL_API float ActivationDerivative(float activation);
	};

	/**
	* �������� ������� ��� �������� ����������� ����������.
	* @return ������ � ��������������.
	*/
	ANNDLL_API std::string GetTestString();

	/**
	* ������� ������ �� �����.
	* @param filepath - ���� � ��� � ����� � �������.
	* @param inputs - ����� ��� ������ ������.
	* @param outputs - ����� ��� ������ �������.
	* @return - ���������� ������.
	*/
	ANNDLL_API bool LoadData(
		std::string filepath,
		std::vector<std::vector<float>> & inputs,
		std::vector<std::vector<float>> & outputs
	);

	/**
	* �������� ������ � ����.
	* @param filepath - ���� � ��� � ����� � �������.
	* @param inputs - ����� ��� ������.
	* @param outputs - ������ ��� ������.
	* @return - ���������� ������.
	*/
	ANNDLL_API bool SaveData(
		std::string filepath,
		std::vector<std::vector<float>> & inputs,
		std::vector<std::vector<float>> & outputs
	);
}
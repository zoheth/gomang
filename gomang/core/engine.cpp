#include "engine.h"

namespace gomang
{

const std::string &IEngine::getName() const
{
	return name_;
}

IEngine::IEngine(std::string model_path, unsigned int num_threads, std::string name) :
    model_path_(std::move(model_path)),
    num_threads_(num_threads),
    name_(std::move(name))
{
}
void IEngine::printTensorInfo() const
{
	const auto input_info  = getInputInfo();
	const auto output_info = getOutputInfo();

	std::cout << "================== Input Info ==================\n";
	for (const auto &desc : input_info)
	{
		desc.print();
	}

	std::cout << "================== Output Info ==================\n";
	for (const auto &desc : output_info)
	{
		desc.print();
	}
}
}        // namespace gomang

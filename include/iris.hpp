
#pragma once
#include <ATen/ATen.h>
#include <torch/torch.h>


namespace custom_models{
namespace datasets{

	class  IRIS : public torch::data::Dataset<IRIS> {
		public:
			enum class Mode { kTrain, kTest };

            explicit IRIS(const std::string& root, Mode mode = Mode::kTrain);

			torch::data::Example<> get(size_t index) override;

			c10::optional<size_t> size() const override;

			bool is_train() const noexcept{
				return is_train_;
			}

			const torch::Tensor& images() const{
				return images_;
			}

			const torch::Tensor& targets() const{
				return targets_;
			}

		private:
			torch::Tensor images_, targets_;
			const bool is_train_;
	};

};
};

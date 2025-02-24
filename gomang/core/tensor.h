#pragma once
#include <string>
#include <vector>

#include "memory.h"

namespace gomang
{

struct TensorDesc
{
	std::vector<int64_t> shape;
	DataType             data_type;
	MemoryLayout         layout;
	MemoryType           mem_type;
	size_t               alignment{64};
	std::string          name;

	[[nodiscard]] size_t getElementsCount() const;
	[[nodiscard]] size_t calculateSize() const;
	void print() const;
};

class ITensor
{
  public:
	virtual void                           *data()       = 0;
	[[nodiscard]] virtual const void       *data() const = 0;
	[[nodiscard]] virtual const TensorDesc &desc() const = 0;
	[[nodiscard]] virtual size_t            size() const = 0;
	virtual ~ITensor()                                   = default;
};

class Tensor : public ITensor
{
  public:
	Tensor(TensorDesc desc, IMemoryAllocator *allocator);

	~Tensor() override;

	[[nodiscard]] const void       *data() const override;
	void                           *data() override;
	[[nodiscard]] const TensorDesc &desc() const override;
	[[nodiscard]] size_t            size() const override;

  private:
	TensorDesc desc_;
	void      *data_{nullptr};

	size_t size_{};

	IMemoryAllocator *allocator_;

	void allocaMemory()
	{
		size_ = desc_.calculateSize();
		if (allocator_)
		{
			data_ = allocator_->allocate(size_, desc_.mem_type);
		}
		else
		{
			data_ = std::aligned_alloc(desc_.alignment, size_);
		}
	}
};
}        // namespace gomang

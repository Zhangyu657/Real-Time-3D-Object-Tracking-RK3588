#ifndef RK3588_DEMO_ENGINE_H
#define RK3588_DEMO_ENGINE_H

#include "types/error.h"
#include "types/datatype.h"
#include <vector>
#include <memory>

class NNEngine
{
public:
    virtual ~NNEngine(){};

    virtual nn_error_e LoadModelFile(const char *model_file) = 0;

    virtual const std::vector<tensor_attr_s> &GetInputShapes() = 0;

    virtual const std::vector<tensor_attr_s> &GetOutputShapes() = 0;

    virtual nn_error_e Run(std::vector<tensor_data_s> &inputs, std::vector<tensor_data_s> &outputs, bool want_float) = 0;
};

std::shared_ptr<NNEngine> CreateRKNNEngine();

#endif // RK3588_DEMO_ENGINE_H

#include "argmaxwithvalue_tiling.h"
#include "register/op_def_registry.h"
#include "kernel_operator.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext* context) {
        // 创建 ArgMaxWithValue 算子的 tiling 数据
        ArgMaxWithValueCustomTilingData tiling;

        // 填充 tiling 数据结构
        const uint32_t BLOCK_DIM = 8;   // 每个 block 的维度
        const uint32_t TILE_NUM = 8;    // tile 数量
        uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
        
        // 获取输入张量的维度信息
        // 正确获取属性值
        std::map<std::string, ge::AttrValue> attrs = context->GetAttrs();
        auto dimensionIt = attrs.find("dimension");
        auto keepDimsIt = attrs.find("keep_dims");

        if (dimensionIt == attrs.end() || keepDimsIt == attrs.end()) {
            // 属性未找到，返回错误
            return ge::GRAPH_FAILED;
        }

        const uint32_t dimension = dimensionIt->second.Get<int32_t>();
        const bool keepDims = keepDimsIt->second.Get<bool>();

        // 设置 tiling 结构中的相关信息
        context->SetBlockDim(BLOCK_DIM);
        tiling.set_totalLength(totalLength);
        tiling.set_tileNum(TILE_NUM);
        tiling.set_dimension(dimension);
        tiling.set_keepDims(keepDims);

        // 将 tiling 数据保存到 buffer 中
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());

        // 设置 tiling 数据的大小
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        // 设置工作空间大小（如果有其他要求的话）
        size_t* currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}

namespace ge {
    static ge::graphStatus InferShape(gert::InferShapeContext* context) {
        // 获取输入和输出形状
        const gert::Shape* x1_shape = context->GetInputShape(0);
        gert::Shape* y_shape = context->GetOutputShape(0);

        // 对于 ArgMaxWithValue 算子，输出形状和输入形状在维度上通常一致
        *y_shape = *x1_shape;

        // 正确获取属性值
        std::map<std::string, ge::AttrValue> attrs = context->GetAttrs();
        auto keepDimsIt = attrs.find("keep_dims");
        auto dimensionIt = attrs.find("dimension");

        if (keepDimsIt == attrs.end() || dimensionIt == attrs.end()) {
            // 属性未找到，返回错误
            return ge::GRAPH_FAILED;
        }

        bool keepDims = keepDimsIt->second.Get<bool>();
        int32_t dimension = dimensionIt->second.Get<int32_t>();

        if (!keepDims) {
            // 移除指定的维度
            // 注意：这里的实现取决于 Shape 类的具体实现
            // 以下代码仅为示例，可能需要根据您的 Shape 类进行调整
            y_shape->RemoveDimension(dimension);
        }

        return ge::GRAPH_SUCCESS;
    }
}

namespace ops {
    class ArgMaxWithValueCustom : public OpDef {
    public:
        explicit ArgMaxWithValueCustom(const char* name) : OpDef(name) {
            // 定义输入和输出
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Output("indice")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Output("values")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
            // 设置推理形状
            this->SetInferShape(InferShape);

            // 设置 Tiling
            this->AICore()
                .SetTiling(optiling::TilingFunc);

            // 配置
            this->AICore().AddConfig("ascend910b");
        }
    };
    OP_ADD(ArgMaxWithValueCustom);
}

extern "C" __global__ __aicore__ void argmaxwithvalue_custom(GM_ADDR x, GM_ADDR indice, GM_ADDR values, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(ArgMaxWithValueCustomTilingData, tiling);

    // 初始化并执行 kernel 操作
    KernelArgMaxWithValue op;
    op.Init(x, indice, values, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
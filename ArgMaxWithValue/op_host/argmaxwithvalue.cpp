#include "argmaxwithvalue_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext* context) {
        // 创建 ArgMaxWithValue 算子的 tiling 数据
        ArgMaxWithValueCustomTilingData tiling;

        // 填充 tiling 数据结构
        const uint32_t BLOCK_DIM = 8;   // 每个 block 的维度
        const uint32_t TILE_NUM = 8;    // tile 数量
        uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
        
        // 获取输入张量的维度信息
        const uint32_t dimension = context->GetAttrs<int32_t>("dimension");
        const bool keepDims = context->GetAttrs<bool>("keep_dims");

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

        // 如果需要考虑 keepDims 属性，可能需要调整输出形状
        bool keepDims = context->GetAttrs<bool>("keep_dims");
        if (!keepDims) {
            // 移除指定的维度
            int32_t dimension = context->GetAttrs<int32_t>("dimension");
            // 这里需要实现 RemoveDimension 或者使用其他方式来移除维度
            // 例如，您可以创建一个新的 Shape 对象，然后复制除了需要移除的维度之外的所有维度
            // 这里只是一个示例，具体实现需要根据您的需求来定
            y_shape->SetShape(x1_shape->GetDims(), x1_shape->GetShapeSize() - 1);
            y_shape->SetDim(dimension, 1); // 将维度设置为1，而不是完全移除
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
            this->AICore().AddConfig("ascend310b");
        }
    };
    // 注册算子
    OP_ADD(ArgMaxWithValueCustom);
}
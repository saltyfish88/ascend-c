#include "register/tilingdata_base.h"

namespace optiling {
    BEGIN_TILING_DATA_DEF(ArgMaxWithValueCustomTilingData)
        // 自定义 tiling 结构体成员变量
        // totalLength：用于描述输入张量的总长度
        TILING_DATA_FIELD_DEF(uint32_t, totalLength);
        // tileNum：表示 tile 的数量
        TILING_DATA_FIELD_DEF(uint32_t, tileNum);
        // dimension：记录 ArgMax 的维度参数
        TILING_DATA_FIELD_DEF(uint32_t, dimension);
        // keepDims：记录是否保持维度
        TILING_DATA_FIELD_DEF(bool, keepDims);
    END_TILING_DATA_DEF;

    // 注册 ArgMaxWithValue 算子相关的 Tiling 数据结构
    REGISTER_TILING_DATA_CLASS(ArgMaxWithValueCustom, ArgMaxWithValueCustomTilingData)
}

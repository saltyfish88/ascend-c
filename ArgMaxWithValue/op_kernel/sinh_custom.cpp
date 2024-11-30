#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelArgMaxWithValue {
public:
    __aicore__ inline KernelArgMaxWithValue() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indice, GM_ADDR values, uint32_t totalLength, uint32_t tileNum)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        ASSERT(tileNum != 0 && "tile num can not be zero!");
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        // 设置全局缓冲区
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * GetBlockIdx(), this->blockLength);
        indiceGm.SetGlobalBuffer((__gm__ DTYPE_INDICE *)indice + this->blockLength * GetBlockIdx(), this->blockLength);
        valuesGm.SetGlobalBuffer((__gm__ DTYPE_VALUES *)values + this->blockLength * GetBlockIdx(), this->blockLength);

        // 初始化队列
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueIndice, BUFFER_NUM, this->tileLength * sizeof(DTYPE_INDICE));
        pipe.InitBuffer(outQueueValues, BUFFER_NUM, this->tileLength * sizeof(DTYPE_VALUES));
        
        // 初始化临时缓冲区
        pipe.InitBuffer(tmpBuffer1, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuffer2, this->tileLength * sizeof(DTYPE_X));
    }

    __aicore__ inline void Process()
    {
        // 定义处理 loopCount
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        // 获取输入和输出
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_INDICE> indiceLocal = outQueueIndice.AllocTensor<DTYPE_INDICE>();
        LocalTensor<DTYPE_VALUES> valuesLocal = outQueueValues.AllocTensor<DTYPE_VALUES>();
        
        // 临时存储最大值和索引
        LocalTensor<DTYPE_X> tmpTensor1 = tmpBuffer1.Get<DTYPE_X>();
        LocalTensor<DTYPE_X> tmpTensor2 = tmpBuffer2.Get<DTYPE_X>();

        DTYPE_X maxVal = -FLT_MAX;  // 初始最大值
        DTYPE_INDICE maxIdx = 0;    // 初始索引

        // 遍历输入值，找到最大值和索引
        for (int32_t i = 0; i < this->tileLength; i++) {
            tmpTensor1[i] = xLocal[i];  // 复制输入数据
            if (tmpTensor1[i] > maxVal) {
                maxVal = tmpTensor1[i];  // 更新最大值
                maxIdx = i;              // 更新最大值的索引
            }
        }

        // 将结果存储到输出
        valuesLocal[0] = maxVal;  // 最大值
        indiceLocal[0] = maxIdx;  // 最大值的索引

        // 将计算结果添加到输出队列
        outQueueIndice.EnQue<DTYPE_INDICE>(indiceLocal);
        outQueueValues.EnQue<DTYPE_VALUES>(valuesLocal);

        // 释放输入数据
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 从输出队列中取出数据并复制到全局内存
        LocalTensor<DTYPE_INDICE> indiceLocal = outQueueIndice.DeQue<DTYPE_INDICE>();
        LocalTensor<DTYPE_VALUES> valuesLocal = outQueueValues.DeQue<DTYPE_VALUES>();
        
        DataCopy(indiceGm[progress * this->tileLength], indiceLocal, this->tileLength);
        DataCopy(valuesGm[progress * this->tileLength], valuesLocal, this->tileLength);

        // 释放输出数据
        outQueueIndice.FreeTensor(indiceLocal);
        outQueueValues.FreeTensor(valuesLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueIndice;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueValues;

    GlobalTensor<half> xGm;
    GlobalTensor<int32_t> indiceGm;
    GlobalTensor<half> valuesGm;

    // 临时缓冲区
    TBuf<QuePosition::VECCALC> tmpBuffer1, tmpBuffer2;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void argmaxwithvalue_custom(GM_ADDR x, GM_ADDR indice, GM_ADDR values, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    
    // 初始化并执行 kernel 操作
    KernelArgMaxWithValue op;
    op.Init(x, indice, values, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}

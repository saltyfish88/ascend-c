#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
class KernelAsinh {
public:
 __aicore__ inline KernelAsinh() {}
 __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t 
tileNum)
 {
 //考生补充初始化代码
 ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
 this->blockLength = totalLength / GetBlockNum();
 this->tileNum = tileNum;
 ASSERT(tileNum != 0 && "tile num can not be zero!");
 this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
 xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * GetBlockIdx(), 
this->blockLength);
 yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * GetBlockIdx(), 
this->blockLength);
 pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
 pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
 pipe.InitBuffer(tmpBuffer1, this->tileLength * sizeof(DTYPE_X));
 pipe.InitBuffer(tmpBuffer2, this->tileLength * sizeof(DTYPE_X));
 pipe.InitBuffer(tmpBuffer3, this->tileLength * sizeof(DTYPE_X));
 pipe.InitBuffer(tmpBuffer4, this->tileLength * sizeof(DTYPE_X));
 pipe.InitBuffer(tmpBuffer5, this->tileLength * sizeof(DTYPE_X));
 pipe.InitBuffer(tmpBuffer6, this->tileLength * sizeof(DTYPE_X));
 }
 __aicore__ inline void Process()
 {
 //考生补充对“loopCount”的定义，注意对 Tiling 的处理
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
 //考生补充算子代码
 LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
 DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
 inQueueX.EnQue(xLocal);
 }
 __aicore__ inline void Compute(int32_t progress)
 {
 //考生补充算子计算代码
 //x>0: asinh(x) = In(x + sqrt(x^2 + 1))
 LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
 LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
 LocalTensor<DTYPE_X> tmpTensor1 = tmpBuffer1.Get<DTYPE_X>();
 LocalTensor<DTYPE_X> tmpTensor2 = tmpBuffer2.Get<DTYPE_X>();
 LocalTensor<DTYPE_X> tmpTensor3 = tmpBuffer3.Get<DTYPE_X>();
 LocalTensor<DTYPE_X> tmpTensor4 = tmpBuffer4.Get<DTYPE_X>();
 LocalTensor<DTYPE_X> tmpTensor5 = tmpBuffer5.Get<DTYPE_X>();
  LocalTensor<DTYPE_X> tmpTensor6 = tmpBuffer6.Get<DTYPE_X>();
 DTYPE_X inputVal1 = 1;
DTYPE_X inputVal2 = -1;
 if(xLocal[0] > =0)
 {
    Muls(tmpTensor1, xLocal, xLocal, this->tileLength);//x^2
    ADD(tmpTensor2, tmpTensor1, inputVal1, this->tileLength);//x^2+1
    Sqrt(tmpTensor3, tmpTensor2, this->tileLength);//  sqrt(x^2+1)
    ADD(tmpTensor4, xLocal, tmpTensor3, this->tileLength);//x+sqrt(x^2+1)
    Log(yLocal, tmpTensor4, this->tileLength);//In(x + sqrt(x^2 + 1))
    outQueueY.EnQue<DTYPE_Y>(yLocal);
    inQueueX.FreeTensor(xLocal);
 }else{
    //x<0: asinh(x) = -In(-x + sqrt(x^2 + 1))
    Muls(tmpTensor1, xLocal, inputVal2, this->tileLength);//-x
    Muls(tmpTensor2, tmpTensor1, tmpTensor1, this->tileLength);//x^2
    ADD(tmpTensor3, tmpTensor2, inputVal1, this->tileLength);//x^2+1
    Sqrt(tmpTensor4, tmpTensor2, this->tileLength);//sqrt(x^2+1)
    ADD(tmpTensor5, tmpTensor1, tmpTensor4, this->tileLength);//-x+sqrt(x^2+1)
    Log(tmpTensor6, tmpTensor5, ,this->tileLength);//In(-x + sqrt(x^2 + 1))
    Muls(yLocal, tmpTensor6, inputVal2, this->tileLength);//-In(-x + sqrt(x^2 + 1))
    outQueueY.EnQue<DTYPE_Y>(yLocal);
    inQueueX.FreeTensor(xLocal);
 }

 }
 __aicore__ inline void CopyOut(int32_t progress)
 {
 //考生补充算子代码
 LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
 DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
 outQueueY.FreeTensor(yLocal);
 }
private:
 TPipe pipe;
 //create queue for input, in this case depth is equal to buffer num
 TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
 //create queue for output, in this case depth is equal to buffer num
 TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
 GlobalTensor<half> xGm;
 GlobalTensor<half> yGm;
 //考生补充自定义成员变量
 TBuf<QuePosition::VECCALC> tmpBuffer1, tmpBuffer2, tmpBuffer3, tmpBuffer4, tmpBuffer5,tmpBuffer6;
 uint32_t blockLength;
 uint32_t tileNum;
 uint32_t tileLength;
};
extern "C" __global__ __aicore__ void Asinh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR 
workspace, GM_ADDR tiling) {
 GET_TILING_DATA(tiling_data, tiling);
 KernelaSinh op;
 //补充 init 和 process 函数调用内容
 op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum);
 op.Process();
}


import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;

import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;

public final class CudaUtil {

    private CudaUtil(){}

    public static CUdeviceptr allocateAndCopy(double[] hostData, int sizeOfElement) {
        CUdeviceptr devicePointer = new CUdeviceptr();
        cuMemAlloc(devicePointer, (long) hostData.length * sizeOfElement);
        cuMemcpyHtoD(devicePointer, Pointer.to(hostData), (long) hostData.length * sizeOfElement);
        return devicePointer;
    }

    public static CUdeviceptr allocateAndCopy(int[] hostData, int sizeOfElement) {
        CUdeviceptr devicePointer = new CUdeviceptr();
        cuMemAlloc(devicePointer, (long) hostData.length * sizeOfElement);
        cuMemcpyHtoD(devicePointer, Pointer.to(hostData), (long) hostData.length * sizeOfElement);
        return devicePointer;
    }

    public static CUdeviceptr allocateAndCopy(byte[] hostData, int sizeOfElement) {
        CUdeviceptr devicePointer = new CUdeviceptr();
        cuMemAlloc(devicePointer, (long) hostData.length * sizeOfElement);
        cuMemcpyHtoD(devicePointer, Pointer.to(hostData), (long) hostData.length * sizeOfElement);
        return devicePointer;
    }

    public static CUdeviceptr allocateAndCopy(float[] hostData, int sizeOfElement) {
        CUdeviceptr devicePointer = new CUdeviceptr();
        cuMemAlloc(devicePointer, (long) hostData.length * sizeOfElement);
        cuMemcpyHtoD(devicePointer, Pointer.to(hostData), (long) hostData.length * sizeOfElement);
        return devicePointer;
    }
}

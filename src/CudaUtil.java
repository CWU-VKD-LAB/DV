import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.runtime.cudaDeviceProp;

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

    public static int getCudaCoresPerSM(int major, int minor) {
        return switch (major) {
            case 2 -> 32;
            case 3 -> 192;
            case 5 -> 128;
            case 6 -> (minor == 1 ? 128 : 64);
            case 7 -> 64;
            case 8 -> (minor == 0 ? 64 : 128);
            case 9 -> 128;
            default -> 2; // Fallback for unknown architectures, assume they at least have cuda cores
        };
    }


    public static int getNumberCudaCores(cudaDeviceProp prop){
        int coresPerSM = getCudaCoresPerSM(prop.major, prop.minor);
        return prop.multiProcessorCount * coresPerSM;
    }
}

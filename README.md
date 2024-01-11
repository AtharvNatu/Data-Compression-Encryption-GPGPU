# Data-Compression-Encryption-GPGPU
A Data Compression and Encryption tool which makes use of GPU to obtain time and power efficiency

### Currently Working Platforms ###

- [x] Windows 11/10 64-bit
- [ ] Linux
- [ ] macOS

## Code Organization ##
1. CPU - Contains Serial C and CPP based code that runs on CPU in single threaded mode
2. CUDA - Contains Parallel code that runs on Nvidia GPUs
3. OpenCL - Contains Parallel code that runs on AMD, Intel and Nvidia GPUs
4. flutter_app - Contains the UI client application code

# Flutter App #
To run the app in Debug Mode, simply run the below command
```shell
flutter run
```

To deploy the app, run the below commands according to your OS

For Windows
```shell
flutter build windows --release
```

For Linux
```shell
flutter build linux --release
```

Copy the **get-gpu-info.sh** and **ffi_lib** directory present inside **lib** to the **build/linux/x64/release/bundle/** directory and then run the executable


![Screenshot 2023-09-23 121813](https://github.com/AtharvNatu/Flutter-GPU-Detection/assets/66716779/b8ff8e60-642a-41a1-8bd1-a08cac3b62f8)

![Screenshot 2023-09-23 121843](https://github.com/AtharvNatu/Flutter-GPU-Detection/assets/66716779/7feb4c79-4eb5-497e-bd89-71f3f6780ade)

![Screenshot 2023-09-23 121855](https://github.com/AtharvNatu/Flutter-GPU-Detection/assets/66716779/54c80232-7bb6-4196-bde7-55b2d871faa5)

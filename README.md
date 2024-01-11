# Data-Compression-Encryption-GPGPU
A Data Compression and Encryption tool which makes use of GPU to obtain time and power efficiency

### Currently Working Platforms ###

- [x] Windows 11/10 64-bit
- [ ] Linux
- [ ] macOS

### Tested GPUs

- [x] AMD Radeon RX6600
- [x] NVIDIA GeForce RTX 3060
- [x] NVIDIA GeForce RTX 3050 Mobile
- [x] NVIDIA GeForce GTX 1650 Mobile
- [x] Intel UHD 630
- [ ] NVIDIA GeForce GTX 1660 Super

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

![Screenshot 2024-01-11 132803](https://github.com/AtharvNatu/Data-Compression-Encryption-GPGPU/assets/66716779/9776bab0-d118-4472-97a8-a5d84d2daa8f)

![2](https://github.com/AtharvNatu/Data-Compression-Encryption-GPGPU/assets/66716779/d555a0b6-e615-42c0-96c3-09cd15ae83c8)

![3](https://github.com/AtharvNatu/Data-Compression-Encryption-GPGPU/assets/66716779/14578cde-25f8-4c81-a3ff-463ae0147d9b)

![4](https://github.com/AtharvNatu/Data-Compression-Encryption-GPGPU/assets/66716779/01b40a9f-efe6-4e74-804f-fa7388f09df3)


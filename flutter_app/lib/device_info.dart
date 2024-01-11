import "package:ffi/ffi.dart";
import "package:win32/win32.dart";
import "dart:io";
import "dart:ffi" as dart_ffi;
import "main.dart";
import "lib.dart";
import 'package:system_info2/system_info2.dart';

const int gb = 1024 * 1024 * 1024;

void getDeviceDetails() {
  // logger.i(
  //     "Platform Info : ${"${SysInfo.operatingSystemName} ${SysInfo.kernelVersion}"}");

  // final cores = SysInfo.cores;
  // logger.i("CPU Name : ${cores[0].name}");

  // if (Platform.isLinux) {
  //   logger.i('Number of Cores    : ${cores.length ~/ 2}');
  //   logger.i('Number of Threads    : ${cores.length}');
  // } else {
  //   logger.i('Number of Cores    : ${cores.length}');
  //   logger.i('Number of Threads    : ${cores.length * 2}');
  // }

  // logger.i('RAM : ${(SysInfo.getTotalPhysicalMemory() / gb).ceil()} GB');

  platformInfo = "${SysInfo.operatingSystemName} ${SysInfo.kernelVersion}";
  final cores = SysInfo.cores;
  cpuName = cores[0].name;
  if (Platform.isLinux) {
    cpuCores = (cores.length ~/ 2).toString();
    cpuThreads = (cores.length).toString();
  } else {
    cpuCores = (cores.length).toString();
    cpuThreads = ((cores.length) * 2).toString();
  }
  totalMemory = (SysInfo.getTotalPhysicalMemory() / gb).ceil().toString();
}

int getGPUDetails() {
  if (Platform.isWindows) {
    return getWinGPUDetails();
  } else if (Platform.isLinux) {
    return getLinuxGPUDetails();
  }
  return 0;
}

int getLinuxGPUDetails() {
  var gpuEnumFile = File('enum-gpu');
  var fileData = gpuEnumFile.readAsLinesSync();
  for (var i = 0; i < fileData.length; i++) {
    gpuList.add(fileData.elementAt(i));
  }
  gpuCount = gpuList.length;
  return gpuCount;
}

int getWinGPUDetails() {
  // Get the initial locator to Windows Management
  final pLoc = IWbemLocator(calloc<COMObject>());

  final clsid = calloc<GUID>()..ref.setGUID(CLSID_WbemLocator);
  final iid = calloc<GUID>()..ref.setGUID(IID_IWbemLocator);

  final proxy = calloc<dart_ffi.Pointer<COMObject>>();
  final pSvc = IWbemServices(proxy.cast());

  final pEnumerator = calloc<dart_ffi.Pointer<COMObject>>();
  IEnumWbemClassObject enumerator;
  final uReturn = calloc<dart_ffi.Uint32>();
  var numGPU = 0;

  // Initialize COM
  var hr = CoInitializeEx(dart_ffi.nullptr, COINIT_MULTITHREADED);
  if (FAILED(hr)) {
    logger.e("CoInitializeEx() Failed !!!", error: "COM Error");
    throw WindowsException(hr);
  }

  // Initialize Security Model
  hr = CoInitializeSecurity(
      dart_ffi.nullptr, // Receive Access Permissions
      -1, // COM negotiates service
      dart_ffi.nullptr, // Authentication services
      dart_ffi.nullptr, // Reserved
      RPC_C_AUTHN_LEVEL_DEFAULT, // Authentication
      RPC_C_IMP_LEVEL_IMPERSONATE, // Impersonation
      dart_ffi.nullptr, // Authentication info
      EOLE_AUTHENTICATION_CAPABILITIES.EOAC_NONE, // Additional capabilities
      dart_ffi.nullptr // Reserved
      );

  if (FAILED(hr)) {
    final winException = WindowsException(hr);
    logger.e("CoInitializeSecurity() Failed : $winException.toString() !!!",
        error: "COM Error");
    CoUninitialize();
    throw winException;
  }

  hr = CoCreateInstance(
      clsid, dart_ffi.nullptr, CLSCTX_INPROC_SERVER, iid, pLoc.ptr.cast());

  if (FAILED(hr)) {
    final winException = WindowsException(hr);
    logger.e("CoCreateInstance() Failed : $winException.toString() !!!",
        error: "COM Error");
    CoUninitialize();
    throw winException;
  }

  hr = pLoc.connectServer(
      TEXT("ROOT\\CIMV2"), // WMI namespace
      dart_ffi.nullptr, // Username
      dart_ffi.nullptr, // User password
      dart_ffi.nullptr, // Locale
      NULL, // Security flags
      dart_ffi.nullptr, // Authority
      dart_ffi.nullptr, // Context object
      proxy // IWbemServices proxy
      );

  if (FAILED(hr)) {
    final winException = WindowsException(hr);
    logger.e("pLoc.connectServer() Failed : $winException.toString() !!!",
        error: "COM Error");
    pLoc.release();
    CoUninitialize();
    throw winException;
  }

  hr = CoSetProxyBlanket(
      proxy.value, // Proxy
      RPC_C_AUTHN_WINNT, // Authentication service
      RPC_C_AUTHZ_NONE, // Authorization service
      dart_ffi.nullptr, // Server principal name
      RPC_C_AUTHN_LEVEL_CALL, // Authentication level
      RPC_C_IMP_LEVEL_IMPERSONATE, // Impersonation level
      dart_ffi.nullptr, // Client identity
      EOLE_AUTHENTICATION_CAPABILITIES.EOAC_NONE // Proxy capabilities
      );

  if (FAILED(hr)) {
    final winException = WindowsException(hr);
    logger.e("CoSetProxyBlanket() Failed : $winException.toString() !!!",
        error: "COM Error");
    pSvc.release();
    pLoc.release();
    CoUninitialize();
    throw winException;
  }

  hr = pSvc.execQuery(
      TEXT("WQL"),
      TEXT("SELECT * FROM Win32_VideoController"),
      WBEM_GENERIC_FLAG_TYPE.WBEM_FLAG_FORWARD_ONLY |
          WBEM_GENERIC_FLAG_TYPE.WBEM_FLAG_RETURN_IMMEDIATELY,
      dart_ffi.nullptr,
      pEnumerator);

  if (FAILED(hr)) {
    final winException = WindowsException(hr);
    logger.e("pSvc.execQuery() Failed : $winException.toString() !!!",
        error: "COM Error");
    pSvc.release();
    pLoc.release();
    CoUninitialize();
    throw winException;
  }

  enumerator = IEnumWbemClassObject(pEnumerator.cast());
  while (enumerator.ptr.address > 0) {
    final pClsObj = calloc<dart_ffi.IntPtr>();

    hr = enumerator.next(
        WBEM_TIMEOUT_TYPE.WBEM_INFINITE, 1, pClsObj.cast(), uReturn);

    if (uReturn.value == 0) break;

    numGPU++;

    final clsObj = IWbemClassObject(pClsObj.cast());

    final vtProp = calloc<VARIANT>();
    hr =
        clsObj.get(TEXT("Name"), 0, vtProp, dart_ffi.nullptr, dart_ffi.nullptr);
    if (SUCCEEDED(hr)) {
      gpuList.add(vtProp.ref.bstrVal.toDartString());
    }

    VariantClear(vtProp);
    free(vtProp);
  }

  pSvc.release();
  pLoc.release();
  CoUninitialize();

  return numGPU;
}

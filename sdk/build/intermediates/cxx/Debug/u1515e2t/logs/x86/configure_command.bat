@echo off
"C:\\Androidstudio\\cmake\\3.18.1\\bin\\cmake.exe" ^
  "-HC:\\Androidstudio\\Camera_java-master\\sdk\\libcxx_helper" ^
  "-DCMAKE_SYSTEM_NAME=Android" ^
  "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON" ^
  "-DCMAKE_SYSTEM_VERSION=21" ^
  "-DANDROID_PLATFORM=android-21" ^
  "-DANDROID_ABI=x86" ^
  "-DCMAKE_ANDROID_ARCH_ABI=x86" ^
  "-DANDROID_NDK=C:\\Androidstudio\\ndk\\23.1.7779620" ^
  "-DCMAKE_ANDROID_NDK=C:\\Androidstudio\\ndk\\23.1.7779620" ^
  "-DCMAKE_TOOLCHAIN_FILE=C:\\Androidstudio\\ndk\\23.1.7779620\\build\\cmake\\android.toolchain.cmake" ^
  "-DCMAKE_MAKE_PROGRAM=C:\\Androidstudio\\cmake\\3.18.1\\bin\\ninja.exe" ^
  "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=C:\\Androidstudio\\Camera_java-master\\sdk\\build\\intermediates\\cxx\\Debug\\u1515e2t\\obj\\x86" ^
  "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=C:\\Androidstudio\\Camera_java-master\\sdk\\build\\intermediates\\cxx\\Debug\\u1515e2t\\obj\\x86" ^
  "-DCMAKE_BUILD_TYPE=Debug" ^
  "-BC:\\Androidstudio\\Camera_java-master\\sdk\\.cxx\\Debug\\u1515e2t\\x86" ^
  -GNinja ^
  "-DANDROID_STL=c++_shared"

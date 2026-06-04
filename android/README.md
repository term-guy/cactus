# Android Build

# Cactus for Android & Kotlin Multiplatform

Run AI models on-device with a simple Kotlin API.

> **Model weights:** Pre-converted weights for all supported models at [huggingface.co/Cactus-Compute](https://huggingface.co/Cactus-Compute).

## Building

<!-- --8<-- [start:install] -->
```bash
git clone https://github.com/cactus-compute/cactus && cd cactus
source ./setup
cactus build --android
```

Build output: `android/libcactus.so` (and `android/libcactus.a`)
<!-- --8<-- [end:install] -->

see the main [README.md](../README.md) for how to use CLI & download weight

### Vendored libcurl (device builds)

To bundle libcurl locally for Android device testing, place artifacts using:

`libs/curl/android/arm64-v8a/libcurl.a` and `libs/curl/include/curl/*.h`

The build auto-detects `libs/curl`. You can override with:

```bash
CACTUS_CURL_ROOT=/absolute/path/to/curl cactus build --android
```

## Integration

<!-- --8<-- [start:integration] -->
### Android-only

1. Copy `libcactus.so` to `app/src/main/jniLibs/arm64-v8a/`
2. Copy `Cactus.kt` to `app/src/main/java/com/cactus/`

### Kotlin Multiplatform

Source files:

| File | Copy to |
|------|---------|
| `Cactus.common.kt` | `shared/src/commonMain/kotlin/com/cactus/` |
| `Cactus.android.kt` | `shared/src/androidMain/kotlin/com/cactus/` |
| `Cactus.ios.kt` | `shared/src/iosMain/kotlin/com/cactus/` |
| `cactus.def` | `shared/src/nativeInterop/cinterop/` |

Binary files:

| Platform | Location |
|----------|----------|
| Android | `libcactus.so` → `app/src/main/jniLibs/arm64-v8a/` |
| iOS | `libcactus-device.a` → link via cinterop |

build.gradle.kts:

```kotlin
kotlin {
    androidTarget()

    listOf(iosArm64(), iosSimulatorArm64()).forEach {
        it.compilations.getByName("main") {
            cinterops {
                create("cactus") {
                    defFile("src/nativeInterop/cinterop/cactus.def")
                    includeDirs("/path/to/cactus/ffi")
                }
            }
        }
        it.binaries.framework {
            linkerOpts("-L/path/to/apple", "-lcactus-device")
        }
    }

    sourceSets {
        commonMain.dependencies {
            implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.0")
        }
    }
}
```
<!-- --8<-- [end:integration] -->

## Usage

```bash
cactus build --android
```

Or directly:

```bash
bash android/build.sh
```

## Output

- `libcactus.so` — Shared library (JNI, for Android apps)
- `libcactus.a` — Static library (for native test binaries)

## Options

| Variable | Default | Description |
|----------|---------|-------------|
| `ANDROID_NDK_HOME` | Auto-detected | Android NDK path |
| `ANDROID_PLATFORM` | `android-21` | Minimum API level |
| `CMAKE_BUILD_TYPE` | `Release` | CMake build type |
| `CACTUS_CURL_ROOT` | `cactus-engine/libs/curl` | Vendored libcurl path |

## Requirements

- Android NDK (install via Android Studio > SDK Tools > NDK)
- CMake 3.10+

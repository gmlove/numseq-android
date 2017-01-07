# TensorFlow Android Number Sequence Recognizer Camera Demo

This is the application for recognize number sequence. 

## Description

This project is based on tensorflow android demo. Unused files are removed to keep it simple.

## To build/install/run

This project should be built under the tensorflow source code, so download 
tensorflow source project before you go ahead. Currently it's built at commit
`e4dde23d58a10c9d0c14005d20d1ecdd599539ac` of tensorflow source code. 

After you clone tensorflow source code, place this project at `//tensorflow/examples/numseq-android`. 

As a prerequisite, Bazel, the Android NDK, and the Android SDK must all be
installed on your system.

1. Get the recommended Bazel version listed at:
        https://www.tensorflow.org/versions/master/get_started/os_setup.html#source
2. The Android NDK may be obtained from:
        http://developer.android.com/tools/sdk/ndk/index.html
3. The Android SDK and build tools may be obtained from:
        https://developer.android.com/tools/revisions/build-tools.html

The Android entries in [`<workspace_root>/WORKSPACE`](../../../WORKSPACE#L2-L13)
must be uncommented with the paths filled in appropriately depending on where
you installed the NDK and SDK. Otherwise an error such as:
"The external label '//external:android/sdk' is not bound to anything" will
be reported.

The TensorFlow `GraphDef` that contains the model definition and weights
is not packaged in the repo because of its size. It can be trained and 
exported by project [num-seq-recognizer](https://github.com/gmlove/num-seq-recognizer).

After editing your WORKSPACE file to update the SDK/NDK configuration,
you may build the APK. Run this from folder `//tensorflow/examples/numseq-android`:

```bash
$ ./gradlew build
```

If you get build errors about protocol buffers, run
`git submodule update --init` from tensorflow root directory and build again.

If adb debugging is enabled on your Android 5.0 or later device, you may then
use the following command from your workspace root to install the APK once
built:

```bash
$ adb install -r -g gradleBuild/outputs/apk/numseq-android-debug.apk
```

Some older versions of adb might complain about the -g option (returning:
"Error: Unknown option: -g").  In this case, if your device runs Android 6.0 or
later, then make sure you update to the latest adb version before trying the
install command again. If your device runs earlier versions of Android, however,
you can issue the install command without the -g option.


If camera permission errors are encountered (possible on Android Marshmallow or
above), then the `adb install` command above should be used instead, as it
automatically grants the required camera permissions with `-g`. The permission
errors may not be obvious if the app halts immediately, so if you installed
with bazel and the app doesn't come up, then the easiest thing to do is try
installing with adb.

Once the app is installed it can be started via the "NumTeller", 
which have the orange TensorFlow logo as their icon.

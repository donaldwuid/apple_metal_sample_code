/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Application entry point for all platforms.
*/

#if defined(TARGET_IOS)
#import <UIKit/UIKit.h>
#import <TargetConditionals.h>
#import "AAPLAppDelegate.h"
#else
#import <Cocoa/Cocoa.h>
#endif

#if defined(TARGET_IOS)

int main(int argc, char *_Nullable *_Nonnull argv)
{
    #if TARGET_OS_SIMULATOR && (!defined(__IPHONE_13_0))
    #error No simulator support for Metal API for this SDK version.
    #endif

    @autoreleasepool
    {
        return UIApplicationMain(argc, argv, nil, NSStringFromClass([AAPLAppDelegate class]));
    }
}

#elif defined(TARGET_MACOS)

int main(int argc, const char *argv[])
{
    return NSApplicationMain(argc, argv);
}

#endif

/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Implementation of the application delegate
*/

#import "AAPLAppDelegate.h"

@implementation AAPLAppDelegate

#if defined(TARGET_IOS)

- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
    return YES;
}

#else

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    return YES;
}

#endif

@end

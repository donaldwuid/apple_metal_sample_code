/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
The header for the cross-platform app delegate.
*/

#import <TargetConditionals.h>

#if !TARGET_OS_IPHONE
#import <Cocoa/Cocoa.h>

@interface AppDelegate : NSObject <NSApplicationDelegate>

@end
#else
#import <UIKit/UIKit.h>

@interface AppDelegate : UIResponder <UIApplicationDelegate>

@property (strong, nonatomic) UIWindow *window;

@end
#endif

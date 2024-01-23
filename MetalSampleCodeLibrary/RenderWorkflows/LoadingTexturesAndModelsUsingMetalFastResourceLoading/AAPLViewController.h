/*
See LICENSE folder for this sample’s licensing information.

Abstract:
The header for the iOS view controller.
*/
#import <TargetConditionals.h>
#import <Cocoa/Cocoa.h>
#import <MetalKit/MetalKit.h>

#define PlatformViewController NSViewController<MTKViewDelegate>

@interface AAPLViewController : PlatformViewController
@end

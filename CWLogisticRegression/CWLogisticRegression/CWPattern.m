//
//  CWPattern.m
//  CWLogisticRegression
//
//  Created by Li Chen wei on 2016/7/10.
//  Copyright © 2016年 TWML. All rights reserved.
//

#import "CWPattern.h"

@interface CWPattern ()


@end

@implementation CWPattern
@synthesize feature,target;

- (instancetype)init
{
    self = [super init];
    if (self) {
        self.feature = [NSMutableArray new];
        self.target = 0;
    }
    return self;
}

- (instancetype)initWithFeature:(NSArray *)features targetValue:(double)targetValue
{
    self = [self init];
    if (self) {
        self.feature = [features mutableCopy];
        self.target = targetValue;
    }
    return self;
}

@end

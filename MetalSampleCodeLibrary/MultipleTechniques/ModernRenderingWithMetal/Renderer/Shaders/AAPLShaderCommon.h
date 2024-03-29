/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Header for types and functions used in multiple Metal shader files.
*/

#import <metal_stdlib>
#import <simd/simd.h>

// Including header shared between this Metal shader code and Swift/C code executing Metal API commands.
#import "AAPLShaderTypes.h"
#import "../../Asset/AAPLMeshTypes.h"

using namespace metal;

// Typedefs to allow quick switching between half and float types for lighting.
typedef half  xhalf;
typedef half2 xhalf2;
typedef half3 xhalf3;
typedef half4 xhalf4;

// Shader representation of an AAPLMaterial.
struct AAPLShaderMaterial
{
    texture2d<xhalf> albedo            [[ id(AAPLMaterialIndexBaseColor) ]];
    texture2d<xhalf> metallicRoughness [[ id(AAPLMaterialIndexMetallicRoughness) ]];
    texture2d<xhalf> normal            [[ id(AAPLMaterialIndexNormal) ]];
    texture2d<xhalf> emissive          [[ id(AAPLMaterialIndexEmissive) ]];
    float alpha                        [[ id(AAPLMaterialIndexAlpha) ]];
    bool hasMetallicRoughness          [[ id(AAPLMaterialIndexHasMetallicRoughness) ]];
    bool hasEmissive                   [[ id(AAPLMaterialIndexHasEmissive) ]];

#if SUPPORT_SPARSE_TEXTURES
    uint baseColorMip           [[ id(AAPLMaterialIndexBaseColorMip) ]];
    uint metallicRoughnessMip   [[ id(AAPLMaterialIndexMetallicRoughnessMip) ]];
    uint normalMip              [[ id(AAPLMaterialIndexNormalMip) ]];
    uint emissiveMip            [[ id(AAPLMaterialIndexEmissiveMip) ]];
#endif
};

#if SUPPORT_SPARSE_TEXTURES
#   define MATERIAL_BASE_COLOR_MIP          (material.baseColorMip)
#   define MATERIAL_METALLIC_ROUGHNESS_MIP  (material.metallicRoughnessMip)
#   define MATERIAL_NORMAL_MIP              (material.normalMip)
#   define MATERIAL_EMISSIVE_MIP            (material.emissiveMip)
#else
#   define MATERIAL_BASE_COLOR_MIP          (0)
#   define MATERIAL_METALLIC_ROUGHNESS_MIP  (0)
#   define MATERIAL_NORMAL_MIP              (0)
#   define MATERIAL_EMISSIVE_MIP            (0)
#endif

// Structure containing light parameters to allow them to be passed to shaders
//  invoked from ICBs.
struct AAPLShaderLightParams
{
    constant AAPLPointLightData *pointLightBuffer               [[ id(AAPLLightParamsIndexPointLights) ]];
    constant AAPLSpotLightData  *spotLightBuffer                [[ id(AAPLLightParamsIndexSpotLights) ]];
    constant uint8_t            *pointLightIndices              [[ id(AAPLLightParamsIndexPointLightIndices) ]];
    constant uint8_t            *spotLightIndices               [[ id(AAPLLightParamsIndexSpotLightIndices) ]];
    constant uint8_t            *pointLightIndicesTransparent   [[ id(AAPLLightParamsIndexPointLightIndicesTransparent) ]];
    constant uint8_t            *spotLightIndicesTransparent    [[ id(AAPLLightParamsIndexSpotLightIndicesTransparent) ]];
};

// Structure containing textures that can be shared globally between shaders.
struct AAPLGlobalTextures
{
    depth2d<float>          viewDepthPyramid    [[ id(AAPLGlobalTextureIndexViewDepthPyramid) ]];

    // Forward rendering needs the following:
    depth2d_array<float>    shadowMap           [[ id(AAPLGlobalTextureIndexShadowMap) ]];

    texture2d<xhalf>        dfgLutTex           [[ id(AAPLGlobalTextureIndexDFG) ]];
    texturecube<xhalf>      envMap              [[ id(AAPLGlobalTextureIndexEnvMap) ]];
    texture2d<float, access::read>  blueNoise   [[ id(AAPLGlobalTextureIndexBlueNoise) ]];
    texture3d<float, access::read>  perlinNoise [[ id(AAPLGlobalTextureIndexPerlinNoise) ]];

#if USE_SCALABLE_AMBIENT_OBSCURANCE
    texture2d<xhalf>        saoTexture          [[ id(AAPLGlobalTextureIndexSAO) ]];
#endif

#if USE_SCATTERING_VOLUME
    texture3d<xhalf>        scattering          [[ id(AAPLGlobalTextureIndexScattering) ]];
#endif

#if USE_SPOT_LIGHT_SHADOWS
    depth2d_array<float>    spotShadowMaps      [[ id(AAPLGlobalTextureIndexSpotShadows) ]];
#endif
};

//----------------------------------------------------------
// Common shader structures
//----------------------------------------------------------

struct AAPLSimpleVertexOut
{
    float4 position [[position]];
};

struct AAPLSimpleTexVertexOut
{
    float4 position [[position]];
    float2 texCoord;
};

//----------------------------------------------------------

// Helper structure to store PBR data.
struct AAPLPixelSurfaceData
{
    xhalf3 normal;
    xhalf3 albedo;
    xhalf3 F0;
    xhalf  roughness;
    xhalf  alpha;
    xhalf3 emissive;
};

// Fragment buffer Gbuffer storage.
struct AAPLGBufferFragOut
{
    xhalf4 albedo      [[color(AAPLGBufferAlbedoAlphaIndex)]];
    xhalf4 normals     [[color(AAPLGBufferNormalsIndex)]];
    xhalf4 emissive    [[color(AAPLGBufferEmissiveIndex)]];
    xhalf4 F0Roughness [[color(AAPLGBufferF0RoughnessIndex)]];
};


#if SUPPORT_DEPTH_PREPASS_TILE_SHADERS

// Declaration of depth data to be stored in image block memory.

struct DepthData
{
    float depth [[color(0)]];
};

// Structure to store depth data in image block memory.
struct TileFragOut
{
    float depth [[color(0)]];
};

#endif // SUPPORT_DEPTH_PREPASS_TILE_SHADERS

//----------------------------------------------------------
// Common Functions
//----------------------------------------------------------

inline float rand(float2 co)
{
    return fract(sin(dot(co.xy ,float2(12.9898,78.233))) * 43758.5453);
}

inline float goldenRatio(float x, int k)
{
    return fract(x + k * (1.0f + sqrt(5.0f)) / 2.0f);
}

//----------------------------------------------------------
//----------------------------------------------------------

inline uint wang_hash (uint seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

inline xhalf3 wang_color (uint seed)
{
    xhalf r = wang_hash(seed)%255/255.0f;
    xhalf g = wang_hash(seed*2)%255/255.0f;
    xhalf b = wang_hash(seed*4)%255/255.0f;

    return xhalf3(r, g, b);
}

//----------------------------------------------------------

static constant uint HEATMAP_LEVELS = 5;

static constant float4 HEATMAP_COLORS[] =
{
    float4(0,0,0,0),
    float4(0,0,1,1),
    float4(0,1,1,1),
    float4(0,1,0,1),
    float4(1,1,0,1),
    float4(1,0,0,1),
};

// Calculates the heatmap color based on a light count for the tile.
inline float4 getHeatmapColor(uint x, uint num)
{
    float l = saturate((float)x / num) * HEATMAP_LEVELS;
    float4 a = HEATMAP_COLORS[(uint)floor(l)];
    float4 b = HEATMAP_COLORS[(uint)ceil(l)];
    float4 heatmap = mix(a, b, l - floor(l));
    return heatmap;
}

//----------------------------------------------------------
// Common Render Functions
//----------------------------------------------------------

// Smoothes the attenuation due to distance for a point or spot light.
inline float smoothDistanceAttenuation(float squaredDistance, float invSqrAttRadius)
{
    float factor = squaredDistance * invSqrAttRadius;
    float smoothFactor = saturate (1.0 - factor * factor);
    return smoothFactor * smoothFactor;
}

// Calculates the attenuation due to distance for a point or spot light.
inline float getDistanceAttenuation(float3 unormalizedLightVector, float invSqrAttRadius)
{
    float sqrDist = dot(unormalizedLightVector, unormalizedLightVector);
    float attenuation = 1.0 / max(sqrDist, 0.01*0.01);
    attenuation *= smoothDistanceAttenuation(sqrDist, invSqrAttRadius);

    return attenuation;
}

// Calculates the attenuation of a spot light due to angle.
//  Float version.
inline float getAngleAttenuation(float3 lightDir, float3 normalizedLightVector, float cosOuter, float cosInner)
{
    float lightAngleScale = 1.0f / max(0.001f, (cosInner - cosOuter));
    float lightAngleOffset = -cosOuter * lightAngleScale;

    float cd = dot(lightDir, normalizedLightVector);
    float attenuation = saturate(cd*lightAngleScale + lightAngleOffset);

    attenuation *= attenuation;
    return attenuation;
}

// Calculates the attenuation of a spot light due to angle.
//  Half version.
inline float getAngleAttenuation(half3 lightDir, half3 normalizedLightVector, float cosOuter, float cosInner)
{
    float lightAngleScale = 1.0f / max(0.001f, (cosInner - cosOuter));
    float lightAngleOffset = -cosOuter * lightAngleScale;

    float cd = dot(lightDir, normalizedLightVector);
    float attenuation = saturate(cd*lightAngleScale + lightAngleOffset);

    attenuation *= attenuation;
    return attenuation;
}

// Converts a 0-1 texture coord on screen and a depth from the depth buffer into
//  a world position by reversing the view projection transform.
inline float4 worldPositionForTexcoord(float2 texCoord, float depth, constant AAPLCameraParams &cameraParams)
{
    float4 ndc;
    ndc.xy = texCoord.xy * 2 - 1;
    ndc.y *= -1;
    ndc.z = depth;
    ndc.w = 1;

    float4 worldPosition = cameraParams.invViewProjectionMatrix * ndc;
    worldPosition.xyz /= worldPosition.w;
    return worldPosition;
}

// Converts a depth from the depth buffer into a view space depth.
inline float linearizeDepth(constant AAPLCameraParams & cameraParams, float depth)
{
    return dot(float2(depth,1), cameraParams.invProjZ.xz) / dot(float2(depth,1), cameraParams.invProjZ.yw);
}

// Generates a shadow value from a world position. Also returns the index of the
//  first cascade containing the world position.
inline float evaluateCascadeShadows(constant AAPLFrameConstants& frameData,
                                    float3                       worldPosition,
                                    depth2d_array<float>         shadowMap,
                                    thread uint&                 cascadeIndex,
                                    bool                         useFilter)
{
    constexpr sampler sam(min_filter::linear, mag_filter::linear, compare_func::less);

    float shadow = 1.0;

    // Figure out which cascade index we're in
    for (cascadeIndex = 0; cascadeIndex < SHADOW_CASCADE_COUNT; cascadeIndex++)
    {
        float4 lightSpacePos = frameData.shadowCameraParams[cascadeIndex].viewProjectionMatrix * float4(worldPosition, 1);
        lightSpacePos /= lightSpacePos.w;

        if (all(lightSpacePos.xyz < 1.0) && all(lightSpacePos.xyz > float3(-1,-1,0)))
        {
            shadow = 0.0f;
            float lightSpaceDepth = lightSpacePos.z - 0.0001f;
            float2 shadowUv = lightSpacePos.xy * float2(0.5, -0.5) + 0.5;

            if(!useFilter)
                return shadowMap.sample_compare(sam, shadowUv, cascadeIndex, lightSpaceDepth);

            for (int j = -1; j <= 1; ++j)
            {
                for (int i = -1; i <= 1; ++i)
                {
                    shadow += shadowMap.sample_compare(sam, shadowUv, cascadeIndex, lightSpaceDepth, int2(i,j));
                }
            }
            shadow /= 9;
            break;
        }
    }
    return shadow;
}

//----------------------------------------------------------
//----------------------------------------------------------

#if USE_SCATTERING_VOLUME

// Converts a view space depth to a depth in scatter volume space.
//  Performs a logarithmic conversion to allow larger froxels in the distance.
inline float zToScatterDepth(float d)
{
    d /= SCATTERING_RANGE;
    d = log2(d * 7.0f + 1.0f) / 3.0f;
    //d = log2(d * 3.0f + 1.0f) / 2.0f;
    return d;
}

// Converts a depth in scatter volume space to a view space depth.
//  Performs a logarithmic conversion to allow larger froxels in the distance.
inline float scatterSliceToZ(float slice)
{
    float d = slice / SCATTERING_VOLUME_DEPTH;
    d = (exp2(d * 3.0f) - 1.0f) * 1.0f/7.0f;
    //d = (exp2(d * 2.0f) - 1.0f) * 1.0f/3.0f;
    return d * SCATTERING_RANGE;
}

inline xhalf4 applyScattering(xhalf4 color, uint2 position, float2 texcoord, float depth,
                              texture3d<xhalf, access::sample> scattering, texture2d<float, access::read> noiseTexture,
                              constant AAPLFrameConstants & frameData, constant AAPLCameraParams & cameraParams)
{
    constexpr sampler linearSampler(mip_filter::linear, mag_filter::linear, min_filter::linear, address::clamp_to_edge);

    float linearDepth = linearizeDepth(cameraParams, depth);
    float scatterDepth = zToScatterDepth(linearDepth);
    scatterDepth = saturate(scatterDepth);

    float2 dither = 0.0f;

    // Small in-place temporal blur
    {
        uint2 ditherPos0 = position;
        uint2 ditherPos1 = position;
        ditherPos1.x += 13;
        ditherPos1.y += 42;

        dither.x = noiseTexture.read(ditherPos0 % 64).x;
        dither.y = noiseTexture.read(ditherPos1 % 64).x;
        dither.x = goldenRatio(dither.x, frameData.frameCounter);
        dither.y = goldenRatio(dither.y, frameData.frameCounter);
        dither = dither * 2.0f - 1.0f;
        dither /= float2(scattering.get_width(), scattering.get_height());

        const float scale = 0.5f;
        dither *= scale;
    }

    xhalf4 scatteringSample = scattering.sample(linearSampler, float3(texcoord.xy + dither, scatterDepth));

    return xhalf4(color.rgb * scatteringSample.a + scatteringSample.rgb * color.a, color.a);
}

inline xhalf3 applyScattering(xhalf3 color, uint2 position, float2 texcoord, float depth,
                              texture3d<xhalf, access::sample> scattering, texture2d<float, access::read> noiseTexture,
                              constant AAPLFrameConstants & frameData, constant AAPLCameraParams & cameraParams)
{
    return applyScattering(xhalf4(color, 1.0f), position, texcoord, depth, scattering, noiseTexture, frameData, cameraParams).rgb;
}

#endif

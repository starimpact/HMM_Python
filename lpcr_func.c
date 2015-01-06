#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "string.h"


typedef unsigned char uchar;


void normalize_img_data_to_0_1(uchar *pubyDataVec, float *pfDataVec, int dwDataLen, int dwRatio)
{
	int dwPI;
	int adwHist[256];
	int adwMinInfo[2], adwMaxInfo[2];
	int dwMinV, dwMaxV;
	float fTmp = 0.f, fTmp2 = 0.f;
	
	memset(adwHist, 0, 256 * sizeof(int));
	for (dwPI = 0; dwPI < dwDataLen; dwPI++)
	{
		adwHist[pubyDataVec[dwPI]]++;
	}
	
	adwMinInfo[0] = 0;
	adwMinInfo[1] = 0;
	for (dwPI = 0; dwPI < 256; dwPI++)
	{
		adwMinInfo[0] += adwHist[dwPI];
		adwMinInfo[1] += adwHist[dwPI] * dwPI;
		if (adwMinInfo[0] * 100 > dwRatio * dwDataLen)
		{
			break;
		}
	}
	
	adwMaxInfo[0] = 0;
	adwMaxInfo[1] = 0;
	for (dwPI = 255; dwPI >= 0; dwPI--)
	{
		adwMaxInfo[0] += adwHist[dwPI];
		adwMaxInfo[1] += adwHist[dwPI] * dwPI;
		if (adwMaxInfo[0] * 100 > dwRatio * dwDataLen)
		{
			break;
		}
	}
	
	dwMinV = adwMinInfo[1] / adwMinInfo[0];
	dwMaxV = adwMaxInfo[1] / adwMaxInfo[0];
	fTmp2 = dwMaxV - dwMinV + 0.00001f;
	for (dwPI = 0; dwPI < dwDataLen; dwPI++)
	{
		fTmp = (pubyDataVec[dwPI] - dwMinV) / fTmp2;
		if (fTmp < 0.f) fTmp = 0.f;
		if (fTmp > 1.f) fTmp = 1.f;
		pfDataVec[dwPI] = fTmp;
	}
}



void imgResize(uchar *patch, int s32W_src, int s32H_src, uchar *result, int s32W_dst, int s32H_dst)
{
    int s32RI, s32CI;
    float srcX, srcY;
    float fSub_X, fSub_Y;
    int s32x, s32y;
    float fCov_X = 1.0 * (s32W_src - 1) / (s32W_dst - 1);
    float fCov_Y = 1.0 * (s32H_src - 1) / (s32H_dst - 1);
    float fTool = 0;
    
    for(s32RI = 0; s32RI < s32H_dst; s32RI++)
    {
        srcY = s32RI * fCov_Y;
        s32y = (int)srcY;
        fSub_Y = srcY - s32y;
        uchar *ptrPatch = patch + s32y * s32W_src;
        uchar *ptrResult = result + s32RI * s32W_dst;
        for(s32CI=0; s32CI < s32W_dst; s32CI++)
        {
            srcX = s32CI * fCov_X;
            s32x = (int)srcX;
            fSub_X = srcX - s32x;
            fTool = fSub_X * fSub_Y;
            if((s32x == s32W_src - 1) || (s32y == s32H_src - 1))
                ptrResult[s32CI] = ptrPatch[s32x];
            else
                ptrResult[s32CI] = (uchar)((1 - fSub_X - fSub_Y + fTool) * ptrPatch[s32x] +
                                           (fSub_Y - fTool) * ptrPatch[s32x + s32W_src] + (fSub_X - fTool) * ptrPatch[s32x + 1] +
                                           fTool * ptrPatch[s32x + s32W_src + 1] + 0.5);
        }
    }
}


int chknegpostag(int adwBBS[4], int *pdwCHBBS, int dwCHBBSLen)
{
    int dwPI;
    int dwPos = 0, dwIllegal = 0;
    int dwH1, dwX1, dwY1, dwX2, dwY2, dwH2, dwW2;
    int dwDistX, dwDistY, dwDistH;
    int dwBBNum;
    int *pdwBBNow = 0;
    int dwTag;
    
    dwBBNum = dwCHBBSLen >> 2;
    dwH1 = adwBBS[3] - adwBBS[1];
    dwX1 = (adwBBS[0] + adwBBS[2]) / 2;
    dwY1 = (adwBBS[1] + adwBBS[3]) / 2;
    for (dwPI = 0; dwPI < dwBBNum; dwPI++) {
        pdwBBNow = pdwCHBBS + dwPI * 4;
        dwX2 = (pdwBBNow[0] + pdwBBNow[2]) / 2;
        dwY2 = (pdwBBNow[1] + pdwBBNow[3]) / 2;
        dwH2 = pdwBBNow[3] - pdwBBNow[1];
        dwW2 = pdwBBNow[2] - pdwBBNow[0];
        dwDistX = abs(dwX1 - dwX2);
        dwDistY = abs(dwY1 - dwY2);
        dwDistH = dwH1 - dwH2;
        
     //   if (dwDistX * 5 <= dwW2 && dwDistY == 0 && dwDistH >= 0 && dwDistH * 6 <= dwH2) {
        if (dwDistX * 5 <= dwW2 && dwDistY == 0 && dwDistH >= 0 && dwDistH * 7 <= dwH1) {
            dwPos = 1;
            break;
        }
        else if (dwDistX * 4 <= dwW2 && dwDistY * 3 <= dwH2 && dwDistH * 3 <= dwH2) {
            dwIllegal = 1;
            break;
        }
    }
    
    dwTag = 0;
    if (dwPos == 1) {
        dwTag = 1;
    }
    else if (dwIllegal == 1){
        dwTag = -1;
    }
    
    return dwTag;
}


int getsamples(uchar *pubyImage, int dwImgW, int dwImgH,
               int *pdwLPBBS, int dwLPBBSLen, int *pdwCHBBS, int dwCHBBSLen, int dwStdW, int dwStdH,
               float *pfSampleList, int *pdwTagList, int *pdwSampleNum)
{
    int dwRI, dwCI, dwBRI, dwBCI;
    int dwSampleNumMax = *pdwSampleNum;
    int adwBBSNow[4];
    float *pfImagePatch = 0;
    uchar *pubyImagePatch = 0;
    int dwSampleNumNow = 0;
    int dwVecLen = dwStdW * dwStdH;
    int dwTag;
    int dwNegStepW = dwImgW / 20, dwNegStepH = dwImgH / 10;
    int dwCentY, dwStdHalfH;
    
    pfImagePatch = (float*)malloc(dwVecLen * sizeof(float));
    pubyImagePatch = (uchar*)malloc(dwVecLen);
    
    dwCentY = (pdwLPBBS[3] + pdwLPBBS[1]) / 2;
    dwStdHalfH = dwStdH / 2;
//    printf("dwImgW:%d, dwImgH:%d, dwStdW:%d, dwStdH:%d, LPBBS:%d,%d,%d,%d\n", dwImgW, dwImgH, dwStdW, dwStdH, pdwLPBBS[0], pdwLPBBS[1], pdwLPBBS[2], pdwLPBBS[3]);
    for (dwRI = 0; dwRI < dwImgH - dwStdH; dwRI += 2) {
        for (dwCI = 0; dwCI < dwImgW - dwStdW; dwCI += 2) {
            adwBBSNow[0] = dwCI;
            adwBBSNow[1] = dwRI;
            adwBBSNow[2] = dwCI + dwStdW;
            adwBBSNow[3] = dwRI + dwStdH;
            
            dwTag = chknegpostag(adwBBSNow, pdwCHBBS, dwCHBBSLen);
            
            if (dwTag == -1) {
                continue;
            }
            
            if (dwTag == 0) {
                if(abs(dwRI + dwStdHalfH - dwCentY) > 2 || dwCI < pdwLPBBS[0] || dwCI > pdwLPBBS[2]) {
                    if (dwRI % dwNegStepH != 0 || dwCI % dwNegStepW != 0 || rand() < 0.7f * RAND_MAX) {
                        continue;
                    }
                }
                else if (rand() < 0.7f * RAND_MAX) {
                    continue;
                }
            }
            
            for (dwBRI = dwRI; dwBRI < dwRI + dwStdH; dwBRI++) {
                memcpy(pubyImagePatch + (dwBRI - dwRI) * dwStdW, pubyImage + dwBRI * dwImgW + dwCI, dwStdW);
            }
            
         //   for (dwBCI = 0; dwBCI < dwVecLen; dwBCI++) {
         //       pfImagePatch[dwBCI] = pubyImagePatch[dwBCI];
         //   }
            
            normalize_img_data_to_0_1(pubyImagePatch, pfImagePatch, dwVecLen, 10);
            
            memcpy(pfSampleList + dwSampleNumNow * dwVecLen, pfImagePatch, dwVecLen * sizeof(float));
            pdwTagList[dwSampleNumNow] = dwTag;
            dwSampleNumNow++;
            
            //add invert sample
            for (dwBRI = 0; dwBRI < dwVecLen; dwBRI++) {
                pfImagePatch[dwBRI] = 1.0 - pfImagePatch[dwBRI];
            }
            
            memcpy(pfSampleList + dwSampleNumNow * dwVecLen, pfImagePatch, dwVecLen * sizeof(float));
            pdwTagList[dwSampleNumNow] = dwTag;
            dwSampleNumNow++;
            
            if (dwSampleNumNow >= dwSampleNumMax) {
                printf("getsamples: over the max sample number.\n");
                break;
            }
        }
        if (dwSampleNumNow >= dwSampleNumMax) {
            break;
        }
    }
    
 //   printf("pdwCHBBS:%d, %d, dwSampleNumNow:%d\n", pdwCHBBS[2]-pdwCHBBS[0], pdwCHBBS[3]-pdwCHBBS[1], dwSampleNumNow);
    
    *pdwSampleNum = dwSampleNumNow;
    
    free(pubyImagePatch);
    free(pfImagePatch);
    
    return 0;
}


int getsamples_scales(uchar *pubyImage, int dwImgW, int dwImgH,
                      int *pdwLPBBS, int dwLPBBSLen, int *pdwCHBBS, int dwCHBBSLen, int dwStdW, int dwStdH,
                      float *pfSampleList, int *pdwTagList, int *pdwSampleNum)
{
#define MAX_SCALES_NUM 20
    int dwI, dwPI;
    float afScales[MAX_SCALES_NUM], fScaleNow;
    uchar *pubyRSZImage = 0;
    int dwRSZImgW, dwRSZImgH;
    int *pdwLPBBS_RSZ = 0;
    int *pdwCHBBS_RSZ = 0;
    int dwSampleNumMax = *pdwSampleNum;
    int dwSampleNumAll = 0, dwSampleNumNow = 0;
    int dwVecLen = dwStdW * dwStdH;
    
    afScales[0] = 1.0f;
    for (dwI = 1; dwI < MAX_SCALES_NUM; dwI++) {
        afScales[dwI] = afScales[dwI-1] * 0.95f;
    }
    
    pubyRSZImage = (uchar*)malloc(dwImgH * dwImgW);
    pdwLPBBS_RSZ = (int*)malloc(dwLPBBSLen * sizeof(int));
    pdwCHBBS_RSZ = (int*)malloc(dwCHBBSLen * sizeof(int));
    
    for (dwI = 0; dwI < MAX_SCALES_NUM; dwI++) {
        fScaleNow = afScales[dwI];
        dwRSZImgW = (int)(dwImgW * fScaleNow + 0.5f);
        dwRSZImgH = (int)(dwImgH * fScaleNow + 0.5f);
        
        if (dwRSZImgH < dwStdH) {
        //    printf("getsamples_scales: over the dwStdH. \n");
            break;
        }
        
        imgResize(pubyImage, dwImgW, dwImgH, pubyRSZImage, dwRSZImgW, dwRSZImgH);
     //   for (dwPI = 0; dwPI < dwRSZImgH; dwPI++) {
      //      memcpy(pubyImage + dwPI * dwImgW, pubyRSZImage + dwPI * dwRSZImgW, dwRSZImgW);
     //   }
     //   break;
        for (dwPI = 0; dwPI < dwLPBBSLen; dwPI++) {
            pdwLPBBS_RSZ[dwPI] = (int)(pdwLPBBS[dwPI] * fScaleNow + 0.5f);
        //    printf("%d, ", pdwLPBBS_RSZ[dwPI]);
        }
        //printf("\n");
        for (dwPI = 0; dwPI < dwCHBBSLen; dwPI++) {
            pdwCHBBS_RSZ[dwPI] = (int)(pdwCHBBS[dwPI] * fScaleNow + 0.5f);
        //    printf("%d, ", pdwCHBBS_RSZ[dwPI]);
        }
        //printf("\n");
        
        dwSampleNumNow = dwSampleNumMax - dwSampleNumAll;
        getsamples(pubyRSZImage, dwRSZImgW, dwRSZImgH, pdwLPBBS_RSZ, dwLPBBSLen, pdwCHBBS_RSZ, dwCHBBSLen,
                   dwStdW, dwStdH, pfSampleList + dwSampleNumAll * dwVecLen, pdwTagList + dwSampleNumAll, &dwSampleNumNow);
        dwSampleNumAll += dwSampleNumNow;
        
        if (dwSampleNumAll >= dwSampleNumMax) {
            printf("getsamples_scales: over maximum sample number.\n");
            break;
        }
    }
    
 //   for (dwPI = 0; dwPI < dwVecLen; dwPI++) {
 //       printf("%.3f, ", pfSampleList[dwPI]);
//    }
//    printf("\n");
    
    *pdwSampleNum = dwSampleNumAll;
    
    free(pubyRSZImage);
    free(pdwLPBBS_RSZ);
    free(pdwCHBBS_RSZ);
    
    return 0;
}


/////////////////////////


int chknegpostag_in_out(int adwBBS[4], int *pdwLPBBS, int dwLPBBSLen)
{
    int dwPI;
    int dwPos = 0, dwIllegal = 0;
    int dwH1, dwW1, dwX1, dwY1, dwX2, dwY2, dwH2, dwW2;
    int dwDistX, dwDistY, dwDistH;
    int dwTag;
    
    dwH1 = (adwBBS[3] - adwBBS[1]) / 2;
    dwW1 = (adwBBS[2] - adwBBS[0]) / 2;
    dwX1 = (adwBBS[0] + adwBBS[2]) / 2;
    dwY1 = (adwBBS[1] + adwBBS[3]) / 2;
    
    dwX2 = (pdwLPBBS[0] + pdwLPBBS[2]) / 2;
    dwY2 = (pdwLPBBS[1] + pdwLPBBS[3]) / 2;
    dwH2 = (pdwLPBBS[3] - pdwLPBBS[1]) / 2;
    dwW2 = (pdwLPBBS[2] - pdwLPBBS[0]) / 2;
    
    dwDistX = abs(dwX1 - dwX2);
    dwDistY = abs(dwY1 - dwY2);
    dwDistH = dwH1 - dwH2;
    
    dwTag = -1;
    
    if (dwDistY <= 1 && dwDistH <= 2 && dwDistH >= 0) {
        if (abs(dwDistX - dwW2) < dwW1 / 4) {
            dwTag = 1;
        }
        else if (abs(dwDistX - dwW2) > dwW1) {
            dwTag = 0;
        }
    }
    
    
    return dwTag;
}


int getsamples_in_out(uchar *pubyImage, int dwImgW, int dwImgH,
               int *pdwLPBBS, int dwLPBBSLen, int dwStdW, int dwStdH,
               float *pfSampleList, int *pdwTagList, int *pdwSampleNum)
{
    int dwRI, dwCI, dwBRI, dwBCI;
    int dwSampleNumMax = *pdwSampleNum;
    int adwBBSNow[4];
    float *pfImagePatch = 0;
    uchar *pubyImagePatch = 0;
    int dwSampleNumNow = 0;
    int dwVecLen = dwStdW * dwStdH;
    int dwTag;
    
    pfImagePatch = (float*)malloc(dwVecLen * sizeof(float));
    pubyImagePatch = (uchar*)malloc(dwVecLen);
    
    //    printf("dwImgW:%d, dwImgH:%d, dwStdW:%d, dwStdH:%d, LPBBS:%d,%d,%d,%d\n", dwImgW, dwImgH, dwStdW, dwStdH, pdwLPBBS[0], pdwLPBBS[1], pdwLPBBS[2], pdwLPBBS[3]);
    for (dwRI = 0; dwRI < dwImgH - dwStdH; dwRI += 2) {
        for (dwCI = 0; dwCI < dwImgW - dwStdW; dwCI += 2) {
            adwBBSNow[0] = dwCI;
            adwBBSNow[1] = dwRI;
            adwBBSNow[2] = dwCI + dwStdW;
            adwBBSNow[3] = dwRI + dwStdH;
            
            dwTag = chknegpostag_in_out(adwBBSNow, pdwLPBBS, dwLPBBSLen);
            
            if (dwTag == -1) {
                continue;
            }
            
            
            for (dwBRI = dwRI; dwBRI < dwRI + dwStdH; dwBRI++) {
                memcpy(pubyImagePatch + (dwBRI - dwRI) * dwStdW, pubyImage + dwBRI * dwImgW + dwCI, dwStdW);
            }
            
            //   for (dwBCI = 0; dwBCI < dwVecLen; dwBCI++) {
            //       pfImagePatch[dwBCI] = pubyImagePatch[dwBCI];
            //   }
            
            normalize_img_data_to_0_1(pubyImagePatch, pfImagePatch, dwVecLen, 10);
            
            memcpy(pfSampleList + dwSampleNumNow * dwVecLen, pfImagePatch, dwVecLen * sizeof(float));
            pdwTagList[dwSampleNumNow] = dwTag;
            dwSampleNumNow++;
            
            
            //add invert sample
            for (dwBRI = 0; dwBRI < dwVecLen; dwBRI++) {
                pfImagePatch[dwBRI] = 1.0 - pfImagePatch[dwBRI];
            }
            
            memcpy(pfSampleList + dwSampleNumNow * dwVecLen, pfImagePatch, dwVecLen * sizeof(float));
            pdwTagList[dwSampleNumNow] = dwTag;
            dwSampleNumNow++;
            
            if (dwSampleNumNow >= dwSampleNumMax) {
                printf("getsamples: over the max sample number.\n");
                break;
            }
        }
        if (dwSampleNumNow >= dwSampleNumMax) {
            break;
        }
    }
    
    //   printf("pdwCHBBS:%d, %d, dwSampleNumNow:%d\n", pdwCHBBS[2]-pdwCHBBS[0], pdwCHBBS[3]-pdwCHBBS[1], dwSampleNumNow);
    
    *pdwSampleNum = dwSampleNumNow;
    
    free(pubyImagePatch);
    free(pfImagePatch);
    
    return 0;
}


int getsamples_scales_in_out(uchar *pubyImage, int dwImgW, int dwImgH,
                      int *pdwLPBBS, int dwLPBBSLen, int dwStdW, int dwStdH,
                      float *pfSampleList, int *pdwTagList, int *pdwSampleNum)
{
#define MAX_SCALES_NUM 20
    int dwI, dwPI;
    float afScales[MAX_SCALES_NUM], fScaleNow;
    uchar *pubyRSZImage = 0;
    int dwRSZImgW, dwRSZImgH;
    int *pdwLPBBS_RSZ = 0;
    int dwSampleNumMax = *pdwSampleNum;
    int dwSampleNumAll = 0, dwSampleNumNow = 0;
    int dwVecLen = dwStdW * dwStdH;
    
    afScales[0] = 1.0f;
    for (dwI = 1; dwI < MAX_SCALES_NUM; dwI++) {
        afScales[dwI] = afScales[dwI-1] * 0.95f;
    }
    
    pubyRSZImage = (uchar*)malloc(dwImgH * dwImgW);
    pdwLPBBS_RSZ = (int*)malloc(dwLPBBSLen * sizeof(int));
    
    for (dwI = 0; dwI < MAX_SCALES_NUM; dwI++) {
        fScaleNow = afScales[dwI];
        dwRSZImgW = (int)(dwImgW * fScaleNow + 0.5f);
        dwRSZImgH = (int)(dwImgH * fScaleNow + 0.5f);
        
        if (dwRSZImgH < dwStdH) {
            //    printf("getsamples_scales: over the dwStdH. \n");
            break;
        }
        
        imgResize(pubyImage, dwImgW, dwImgH, pubyRSZImage, dwRSZImgW, dwRSZImgH);
        
        for (dwPI = 0; dwPI < dwLPBBSLen; dwPI++) {
            pdwLPBBS_RSZ[dwPI] = (int)(pdwLPBBS[dwPI] * fScaleNow + 0.5f);
            //    printf("%d, ", pdwLPBBS_RSZ[dwPI]);
        }
        
        dwSampleNumNow = dwSampleNumMax - dwSampleNumAll;
        getsamples_in_out(pubyRSZImage, dwRSZImgW, dwRSZImgH, pdwLPBBS_RSZ, dwLPBBSLen,
                   dwStdW, dwStdH, pfSampleList + dwSampleNumAll * dwVecLen, pdwTagList + dwSampleNumAll, &dwSampleNumNow);
        dwSampleNumAll += dwSampleNumNow;
        
        if (dwSampleNumAll >= dwSampleNumMax) {
            printf("getsamples_scales: over maximum sample number.\n");
            break;
        }
    }
    
    //   for (dwPI = 0; dwPI < dwVecLen; dwPI++) {
    //       printf("%.3f, ", pfSampleList[dwPI]);
    //    }
    //    printf("\n");
    
    *pdwSampleNum = dwSampleNumAll;
    
    free(pubyRSZImage);
    free(pdwLPBBS_RSZ);
    
    return 0;
}




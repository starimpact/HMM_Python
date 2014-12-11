#include "stdio.h"
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


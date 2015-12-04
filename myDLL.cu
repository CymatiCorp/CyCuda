#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <assert.h>
#include <string>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

using namespace std;

typedef struct {

  DWORD  mVersion; //!< mIRC Version
  HWND   mHwnd;    //!< mIRC Hwnd
  BOOL   mKeep;    //!< mIRC variable stating to keep DLL in memory

} LOADINFO;

typedef struct {

  HANDLE m_hFileMap; //!< Handle to the mIRC DLL File Map
  LPSTR m_pData;     //!< Pointer to a character buffer of size 900 to send mIRC custom commands
  HWND m_mIRCHWND;   //!< mIRC Window Handle

} mIRCDLL;


mIRCDLL mIRCLink;

//Initialize Variables.
int noFrozen = 1;       // Prevent mIRC from freezing.


void WINAPI LoadDll(LOADINFO * load) {
  mIRCLink.m_hFileMap = CreateFileMapping( INVALID_HANDLE_VALUE, 0, PAGE_READWRITE, 0, 4096, "mIRC" );     
  mIRCLink.m_pData = (LPSTR) MapViewOfFile( mIRCLink.m_hFileMap, FILE_MAP_ALL_ACCESS, 0, 0, 0 );
  mIRCLink.m_mIRCHWND = load->mHwnd;
}

int WINAPI UnloadDll( int timeout ) {

  // DLL unloaded because mIRC exits or /dll -u used
  if ( timeout == 0 ) {
    UnmapViewOfFile( mIRCLink.m_pData );
    CloseHandle( mIRCLink.m_hFileMap );
    return 1;
  }
  // Keep DLL In Memory
  else
    return 0;
}

int __declspec(dllexport) __stdcall cudaCard(HWND mWnd, HWND aWnd, char *data, char *parms, BOOL show, BOOL nopause) {
    
    	// Prevents mIRC from Freezing.
    if (noFrozen == 1) {
	 MSG msg;
	 PeekMessage(&msg,NULL,0,0,PM_REMOVE);
	 TranslateMessage(&msg);
	 DispatchMessage(&msg);
	}
    
    int nDevices;
    string str = "";

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
     cudaDeviceProp prop;
     cudaGetDeviceProperties(&prop, i);
     str = str + prop.name + " @" ; 
    }
    char *cstr = &str[0u];
    strcpy(data, cstr);
  return 3;
}

// Enable/Disable "No Freeze" Function.
int __declspec(dllexport) __stdcall noFreeze(HWND mWnd, HWND aWnd, char *data, char *parms, BOOL show, BOOL nopause) {
 if (noFrozen == 1) {
  noFrozen = 0;
  strcpy(data, "disabled");
 } else {
  noFrozen = 1;
  strcpy(data, "enabled");
 }
 return 3;
}


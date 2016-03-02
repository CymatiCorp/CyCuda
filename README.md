# CyCuda

Cuda (32-bit) for mIRC
--------------

Create a VC++ DLL in Visual Studio 2012 for mIRC with CUDA<br><br><br>
Requirements:<br>
* Visual Studio 2012 (Not Community)<br> 
* CUDA Toolkit v7.5 https://developer.nvidia.com/cuda-toolkit <br> 

For a build in VS 2013 Community, please reference<br>
https://github.com/mryanbrown/mCUDA 
<br>

Note: Install VS before CUDA toolkit.
<br><br>
<img src='http://cymaticorp.com/cycuda/1.png' width=80% height=80%></img> <br>
Create New Project File . . . <br><br>

<img src='http://cymaticorp.com/cycuda/2.png' width=80% height=80%></img> <br>
Installed > Templates > Visual C++ > Win32 > Create Win32 Project <br>
Name your Project (ie: myDLL) <br>
Press "OK"<br><br>

<img src='http://cymaticorp.com/cycuda/3.png' width=80% height=80%></img> <br>
Check (x) DLL<br>
Check [x] Empty Project<br><br>

<img src='http://cymaticorp.com/cycuda/4.png' width=80% height=80%></img> <br>
Right Click your solution workspace name "myDLL"<br>
Click "Properties"<br><br>

<img src='http://cymaticorp.com/cycuda/c1.png' width=80% height=80%></img> <br>
Change "Character Set" to "Use Multi-Byte Character Set"<br>
Note: for CUDA to work, the "Platform Toolset" should be v110 which is<br>
why we're using VS2012.<br><br>

<img src='http://cymaticorp.com/cycuda/c2.png' width=80% height=80%></img> <br>
In the same properties menu, navigate down to<br>
"VC++ Directories" and ensure the following directories are included:<br>

Include Directories: <br>
```
$(VCInstallDir)include;$(VCInstallDir)atlmfc\include;$(WindowsSDK_IncludePath);C:\ProgramData\NVIDIA Corporation\CUDA Samples\v7.5\common\inc\; 
```
<BR><BR>


<img src='http://cymaticorp.com/cycuda/c3.png' width=80% height=80%></img> <br><br>
Library Directories:<br>
```
$(VCInstallDir)lib;$(VCInstallDir)atlmfc\lib;$(WindowsSDK_LibraryPath_x86);C:\ProgramData\NVIDIA Corporation\CUDA Samples\v7.5\common\lib\; 
```
<br><br>

<img src='http://cymaticorp.com/cycuda/c4.png' width=80% height=80%></img> <br><br>
Navigate to Linker > General > Additional Library Directories:
```
%(AdditionalLibraryDirectories);$(CudaToolkitLibDir);$(CUDA_LIB_PATH);C:\ProgramData\NVIDIA Corporation\CUDA Samples\v7.5\common\lib\; 
```
<br><br>

<img src='http://cymaticorp.com/cycuda/c5.png' width=80% height=80%></img> <br>
Navigate to Linker > Input<br>
Add "cudart.lib" to the beginning, so it looks like this:<br>
```
cudart.lib;kernel32.lib;user32.lib;gdi32.lib;winspool.lib;comdlg32.lib;advapi32.lib;shell32.lib;ole32.lib;oleaut32.lib;uuid.lib;odbc32.lib;odbccp32.lib;%(AdditionalDependencies) 
```

<br><br>
Apply these settings and press "OK"<br><Br>
<img src='http://cymaticorp.com/cycuda/b3.png' width=80% height=80%></img> <br>
Right Click your solution workspace "myDLL"<br>
Navigate to "Build Customizations..."<br>
<img src='http://cymaticorp.com/cycuda/b4.png' width=80% height=80%></img> <br>
Now to add the CUDA dependancies click [X] CUDA 7.5(.targets, .props)<BR>
Press "OK"<br><br>
<img src='http://cymaticorp.com/cycuda/b1.png' width=80% height=80%></img> <br>
Right click your solutions "Source Code" section<br>
Click Add > New Item...<br><br>
<img src='http://cymaticorp.com/cycuda/b2.png' width=80% height=80%></img> <br>
Navigate to "NVIDIA CUDA 7.5" > Code <br>
Select "CUDA C/C++ File"<br>
Name it, the same as your DLL name. (ie: myDLL)<br>
Press "Add"<br><br>
<img src='http://cymaticorp.com/cycuda/d1.png' width=80% height=80%></img> <br>
Right click your solutions "Source Files" section.<br>
Click Add > New Item...<br><br>

<img src='http://cymaticorp.com/cycuda/d2.png' width=80% height=80%></img> <br>
Navigate to Installed > Visual C++ > Code <br>
Select "Module-Definition File (.def) <Br>
Name it, the same as your DLL name. (ie: myDLL) <br>
Press "Add"<br><br>

<img src='http://cymaticorp.com/cycuda/d3.png' width=80% height=80%></img> <br>
Paste Code into myDLL.cu<br>
```
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


```
<br><br>
<img src='http://cymaticorp.com/cycuda/d4.png' width=80% height=80%></img> <br>
Paste Code into myDLL.def<br>
```
LIBRARY myDLL
EXPORTS
cudaCard
LoadDll
UnloadDll
```
<br><br>
<img src='http://cymaticorp.com/cycuda/d5.png' width=80% height=80%></img> <br>

Build Solution (F7) or Rebuild Solution.<br>
Your output should look similar.<br><br>
<img src='http://cymaticorp.com/cycuda/e1.png' width=80% height=80%></img> <br>
Open mIRC and test the DLL. <br>
with //echo -a $dll(myDLL.dll,cudaCard,$null)<br>
the DLL will remain loaded until you unload it with /dll -u myDLL.dll<br>
<br><br><br>
Note:<br>
I did have one compile where the "Build Succeeded" however there was no DLL file.<br>
and my lastbuildstate kept showing up. #v4.0:v110:false <BR>

What I did to reset it was:<br>
* Remove the .CU file.
* Add a regular VC++ CPP file.<BR>
 (using  the same code that was in the .CU) 
* Compile. 
* Add a .CU file again (with same code).
* Compile again (with CPP file)
* Delete  .CPP file from the project.<BR><br>
That worked for me. Let me know if you have issues.<BR>
<br><br>
Another thing, this is a 32-bit DLL, that means that CUDA has to compile in 32-bit mode.<br>
I have not tested the extent of the limitations, but from what I have read there are some.<br>
I tried a 64-bit Compile and mIRC didn't recognize it. I did read that mIRC has been <br>
porting a lot of its code over in expectations to becoming 64-bit, so perhaps it might not<br>
be too long before it might be accepted.<br>

```
Update December 2, 2015
* Added new code to prevent mIRC from freezing when too many calls are made to it.
   Simply call the $dll(myDLL,noFreeze) to enable/disable this feature.
```

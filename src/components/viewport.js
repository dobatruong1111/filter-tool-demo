import { useEffect, useState, useRef } from 'react';
import './viewport.css';

// Load the rendering pieces we want to use (for both WebGL and WebGPU)
import '@kitware/vtk.js/Rendering/Profiles/All';

// Force DataAccessHelper to have access to various data source
import '@kitware/vtk.js/IO/Core/DataAccessHelper/HtmlDataAccessHelper';
import '@kitware/vtk.js/IO/Core/DataAccessHelper/HttpDataAccessHelper';
import '@kitware/vtk.js/IO/Core/DataAccessHelper/JSZipDataAccessHelper';

import vtkGenericRenderWindow from '@kitware/vtk.js/Rendering/Misc/GenericRenderWindow';
import vtkHttpDataSetReader from '@kitware/vtk.js/IO/Core/HttpDataSetReader';
import vtkImageMapper from '@kitware/vtk.js/Rendering/Core/ImageMapper';
import vtkImageSlice from '@kitware/vtk.js/Rendering/Core/ImageSlice';
import vtkInteractorStyleManipulator from '@kitware/vtk.js/Interaction/Style/InteractorStyleManipulator';
import vtkITKHelper from '@kitware/vtk.js/Common/DataModel/ITKHelper';

import readImageDICOMFileSeries from 'itk/readImageDICOMFileSeries';
import readImageFile from 'itk/readImageFile';
import itkConfig from "itk/itkConfig";

itkConfig.itkModulesPath = "/itk";

function Viewport() {
    const vtkContainerRef = useRef(null);
    const context = useRef(null);
    const [file, setFile] = useState();

    const handleLoadFile = (e) => {
        setFile(e.target.files);
    }

    const uniformArray = (len, value) => {
        let array = [];
        for (let i = 0; i < len; ++i) {
            array.push(Array.isArray(value) ? [...value] : value)
        }
        return array
    }

    const convolution = (kernel, matrix) => {
        const rows = matrix.length;
        const cols = matrix[0].length;
        const result = uniformArray(rows, uniformArray(cols, 0));
        const kRows = kernel.length;
        const kCols = kernel[0].length;
        // find center position of kernel (half of kernel size)
        const kCenterX = Math.floor(kCols / 2);
        const kCenterY = Math.floor(kRows / 2);
        let i, j, m, n, ii, jj;
        for (i = 0; i < rows; ++i) {
            for (j = 0; j < cols; ++j) {
                for (m = 0; m < kRows; ++m) {
                    for (n = 0; n < kCols; ++n) {
                        // index of input signal, used for checking boundary
                        ii = i + (m - kCenterY);
                        jj = j + (n - kCenterX);
                        // ignore input samples which are out of bound
                        if (ii >= 0 && ii < rows && jj >= 0 && jj < cols) {
                            result[i][j] += matrix[ii][jj] * kernel[m][n];
                        }
                    }
                }
            }
        }
        return result;
    }

    const filter = (imageData, kernel, slice = 0) => {
        const dims = imageData.getDimensions();
        const s = dims[0] * dims[1];

        const scalars = imageData.getPointData().getScalars(); // vtkDataArray

        // Get pixel data
        const temp = [];
        for (let i = s * slice; i < s * slice + s; ++i) {
            temp.push(scalars.getValue(i));
        }

        // Convert to matrix
        const width = dims[1];
        const matrix = temp.reduce((rows, key, index) => {
            return (index % width == 0 ? rows.push([key]) : rows[rows.length - 1].push(key)) && rows;
        }, []);

        // Filter
        const result = convolution(kernel, matrix);

        // Convert to array
        const array = result.reduce((prev, next) => prev.concat(next));

        let ii = 0;
        for (let i = s * slice; i < s * slice + s; ++i) {
            scalars.setValue(i, array[ii]);
            ii += 1;
        }
    }

    useEffect(() => {
        if (!context.current) {
            const genericRenderWindow = vtkGenericRenderWindow.newInstance({
                background: [0, 0, 0],
            });
            genericRenderWindow.setContainer(vtkContainerRef.current);
            genericRenderWindow.resize();

            // Pipeline
            const reader = vtkHttpDataSetReader.newInstance({ fetchGzip: true });
            const imageMapper = vtkImageMapper.newInstance();
            const imageActor = vtkImageSlice.newInstance();
            const renderer = genericRenderWindow.getRenderer();
            const renderWindow = genericRenderWindow.getRenderWindow();
            const interactorStyle = vtkInteractorStyleManipulator.newInstance();
            interactorStyle.removeAllMouseManipulators();

            // Setup camera
            const camera = renderer.getActiveCamera();
            camera.setParallelProjection(true);

            // Setup render window
            renderWindow.addRenderer(renderer);
            renderWindow.getInteractor().setInteractorStyle(interactorStyle);

            // Init kernels
            // Sharpening filter
            // const kernel = [
            //     [0, -1, 0],
            //     [-1, 5, -1],
            //     [0, -1, 0]
            // ];
            // Blurring filter
            // const kernel = [
            //     [1 / 9, 1 / 9, 1 / 9],
            //     [1 / 9, 1 / 9, 1 / 9],
            //     [1 / 9, 1 / 9, 1 / 9]
            // ];
            // Embossing filter
            // const kernel = [
            //     [-2, -1, 0],
            //     [-1, 1, 1],
            //     [0, 1, 2]
            // ];
            // Edge detection filter
            const kernel = [
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ];

            if (file) {
                if (file.length === 1) {
                    readImageFile(null, file[0])
                        .then(function ({ image, webWorker }) {
                            webWorker.terminate();
                            const imageData = vtkITKHelper.convertItkToVtkImage(image); // vtkImageData
                            filter(imageData, kernel);

                            imageMapper.setInputData(imageData);
                            imageMapper.setKSlice(0);

                            imageActor.setMapper(imageMapper);

                            renderer.addActor(imageActor);
                            renderer.resetCamera();
                            renderer.resetCameraClippingRange();
                            renderWindow.render();
                        })
                        .catch(error => {
                            console.error(error);
                        })
                } else {
                    readImageDICOMFileSeries(null, file)
                        .then(function ({ image, webWorker }) {
                            webWorker.terminate();
                            const imageData = vtkITKHelper.convertItkToVtkImage(image); // vtkImageData
                            const slice = 0;
                            filter(imageData, kernel, slice);

                            imageMapper.setInputData(imageData);
                            imageMapper.setKSlice(slice);

                            imageActor.setMapper(imageMapper);

                            renderer.addActor(imageActor);
                            renderer.resetCamera();
                            renderer.resetCameraClippingRange();
                            renderWindow.render();
                        })
                        .catch(error => {
                            console.error(error);
                        })
                }
            }

            context.current = {
                genericRenderWindow,
                renderer,
                reader,
                imageActor,
                imageMapper
            };
        }

        // Cleanup function
        return () => {
            if (context.current) {
                const {
                    genericRenderWindow,
                    reader,
                    imageActor,
                    imageMapper
                } = context.current;
                imageActor.delete();
                imageMapper.delete();
                reader.delete();
                genericRenderWindow.delete();
                context.current = null;
            };
        };
    }, [vtkContainerRef, file]);

    return (
        <>
            <div>
                <label>Select DICOM file series or DICOM file:</label>
                <input name="inputFile" type="file" onChange={handleLoadFile} multiple />
            </div>
            <div className='root'>
                <div className='view1'>
                    <div className='viewport' ref={vtkContainerRef} />
                </div>
            </div>
        </>
    );
}

export default Viewport;

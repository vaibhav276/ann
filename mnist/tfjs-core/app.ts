import * as mnist from 'mnist';
import * as tf from '@tensorflow/tfjs';
import { activation } from '@tensorflow/tfjs-layers/dist/exports_layers';
import { metrics } from '@tensorflow/tfjs';
import * as $ from 'jquery';
import * as _ from 'underscore';
import * as cjs from 'chart.js';

type ModelShape = {
    input: {
        units: number,
        shape: [number]
    },
    hidden: [
        {
            units: number
        }
    ],
    output: {
        units: number
    }
}

type HyperParams = {
    training: {
        learningRate: number
    },
    output: {
        activation: string
    },
    loss: string,
    metrics: [string],
    batchSize: number,
    epochs: number,
    testFrequency: number
}

type TrainingOutput = {
    loss: number,
    accuracy: number
}

type DataPoint = {
    input: [number],
    output: [number]
}

type Data = [DataPoint];

type PlotData = [
    {
        x: number,
        y: number
    }
]

function createModel(modelShape: ModelShape,
                     hyperParams: HyperParams): tf.Model {
    let model = tf.sequential();

    model.add(tf.layers.dense({
        units: modelShape.input.units,
        inputShape: modelShape.input.shape
    }));

    model.add(tf.layers.dense({
        units: modelShape.hidden[0].units
    }));

    // output layer
    model.add(tf.layers.dense({
        units: modelShape.output.units,
        activation: hyperParams.output.activation
    }));

    let optimizer = tf.train.sgd(hyperParams.training.learningRate);

    model.compile({
        optimizer: optimizer,
        loss: hyperParams.loss,
        metrics: hyperParams.metrics
    });

    return model;
}

async function trainModel(hyperParams: HyperParams,
                          modelShape: ModelShape,
                          model: tf.Model,
                          trainingData: Data,
                          testData: Data): Promise<void> {

    console.log('Training model...');

    const trainingDataChunks = <[Data]>_.chunk(trainingData, hyperParams.batchSize);

    for (let i = 0; i < trainingDataChunks.length; i++) {
        const trainingDataT = toTensors(trainingDataChunks[i]);
        const res = await model.fit(trainingDataT.xs,
                                    trainingDataT.labels,
                                    {
                                        batchSize: trainingDataChunks[i].length,
                                        epochs: hyperParams.epochs,
                                        validationSplit: 0.8
                                    }
                                   );

        // console.log('batch #' + i +' loss = ' + res.history.loss[0]
        //            + ', accuracy = ' + res.history.acc[0]);
        tf.dispose(trainingDataT);
        await tf.nextFrame();

        updatePlot(lossPlot, i, res.history.loss[0] as number);
        updatePlot(accPlot, i, 100*(res.history.acc[0] as number));

        if (i % hyperParams.testFrequency == 0) {
            const evalAcc = evaluateModel(modelShape,
                                          model,
                                          testData);
            await tf.nextFrame();

            updatePlot(evalPlot, i, evalAcc * 100);
        }
    }
}

function evaluateModel(modelShape: ModelShape,
                       model: tf.Model,
                       testData: Data): number {

    console.log('Evaluating model...');

    const testDataT = toTensors(testData);

    const predictionsT = model.predict(testDataT.xs, {
        batchSize: 100
    });

    let predictions = <number[][]>_.chunk(predictionsT.dataSync(),
                                modelShape.output.units);
    predictions = _.map(predictions, arr => {
        let max = _.max(arr);
        return _.map(arr, e => {
            return (e == max ? 1 as number : 0 as number);
        });
    })

    tf.dispose(testDataT);
    tf.dispose(predictionsT);

    let comparison = _.zip(predictions, _.pluck(testData, 'output'));
    // console.log(comparison);

    let correctCount = _.reduce(comparison, (count, arrs) => {
        return (count + (_.isEqual(arrs[0], arrs[1]) ? 1 : 0));
    }, 0);
    console.log('correct/total = %d/%d', correctCount, testData.length);

    return correctCount / testData.length;
}

// NOTE: client must dispose the returned tensors after use
function toTensors(data: Data): {
    xs: tf.Tensor,
    labels: tf.Tensor
} {
    let xsArr = <[number]>_.pluck(data, 'input');
    let labelsArr = <[number]>_.pluck(data, 'output');

    let xsT = tf.tensor2d(xsArr);
    let labelsT = tf.tensor2d(labelsArr);

    return {
        xs: xsT,
        labels: labelsT
    }
}

function initPlot(elem: string, label: string, color: string, title: string): cjs.Chart {

    const lCtx = document.getElementById(elem);
    let plot = new cjs.Chart(lCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: label,
                fill: false,
                borderColor: color,
                data: []
            }]
        },
        options: {
				    responsive: true,
				    title: {
					      display: true,
					      text: title
				    },
				    tooltips: {
					      mode: 'index',
					      intersect: false,
				    },
				    hover: {
					      mode: 'nearest',
					      intersect: true
				    },
				    scales: {
					      xAxes: [{
						        display: true,
						        scaleLabel: {
							          display: true,
							          labelString: 'Batch'
						        }
					      }],
					      yAxes: [{
						        display: true,
						        scaleLabel: {
							          display: true,
							          labelString: 'Value'
						        }
					      }]
				    }
			  }
    });

    return plot;
}

function updatePlot(chart: cjs.Chart, x: number, y:number): void {
    chart.data.labels.push(x);
    chart.data.datasets[0].data.push(y);
    chart.update({
        duration: 0
    });
}

function enableFreehandInput(): void {
    let clicking = false;
    let canvas = $('#freehand')[0] as HTMLCanvasElement;
    let ctx = canvas.getContext('2d');
    let prevX = 0;
    let prevY = 0;
    let currX = 0;
    let currY = 0;
    let dotFlag = false;
    let fillStyle = 'white';

    $('#manual').show();

    $('#freehand').mousedown(e => {
        prevX = currX;
        prevY = currY;
        currX = e.clientX - canvas.offsetLeft;
        currY = e.clientY - canvas.offsetTop;
        clicking = true;
        dotFlag = true;
        if (dotFlag) {
            ctx.beginPath();
            ctx.fillStyle = fillStyle;
            ctx.fillRect(currX, currY, 20, 20);
            ctx.closePath();
            dotFlag = false;
        }
    });
    $('#freehand').mouseup(e => {
        clicking = false;
    });
    $('#freehand').mouseout(e => {
        clicking = false;
    });
    $('#freehand').mousemove(e => {
        if (clicking) {
            prevX = currX;
            prevY = currY;
            currX = e.clientX - canvas.offsetLeft;
            currY = e.clientY - canvas.offsetTop;

            ctx.beginPath();
            ctx.moveTo(prevX, prevY);
            ctx.lineTo(currX, currY);
            ctx.strokeStyle = fillStyle;
            ctx.lineWidth = 20;
            ctx.stroke();
            ctx.closePath();
        }
    });
}

$('#predict').click(() => {
    let canvas = $('#freehand')[0] as HTMLCanvasElement;
    let ctx = canvas.getContext('2d');
    const imgd = ctx.getImageData(0, 0, 140, 140);
    const pix = imgd.data;

    // let cout = document.createElement('canvas');
    let cout = $('#lowres')[0] as HTMLCanvasElement;
    let octx = cout.getContext('2d');
    octx.clearRect(0, 0, 28, 28);
    octx.drawImage(canvas, 0, 0, 140, 140, 0, 0, 28, 28);

    const imgd1 = octx.getImageData(0, 0, 28, 28);
    const pix1 = imgd1.data;

    // $(cout).remove();

    let pix2 = []
    for (let i = 0; i < pix1.length; i++) {
        if (i % 4 == 0) {
            let j = i / 4;
            pix2[j] = pix1[i];
            pix2[j+1] = 0;
            pix2[j+2] = 0;
            pix2[j+3] = 255;
        }
    }
    pix2 = _.take(pix2, 784);

    let res = predictSingleInput(pix2);
    let resNum = arrToNum(res);
    $('#prediction').html(resNum.toString());
});

$('#clear').click(() => {
    let canvas = $('#freehand')[0] as HTMLCanvasElement;
    let ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, 140, 140);
    $('#prediction').html('?');
});

function predictSingleInput(pix: number[]): number[] {
    const pixT = tf.tensor2d(pix, [1, 784]);
    const resT = globalModel.predict(pixT);
    const resArr = resT.dataSync();
    tf.dispose(pixT);
    tf.dispose(resT);

    let max = _.max(resArr);
    const res = _.map(resArr, e => {
        return (e == max ? 1 as number : 0 as number);
    });

    // console.log(res);

    return res;
}

function arrToNum(arr: number[]): number {
    for(let i = 0; i < arr.length; i++) {
        if (arr[i] == 1) {
            return i;
        }
    }
}

function main(): void {
    let fullset = mnist.set(8000, 2000);
    let trainingData = fullset.training;
    let testData = fullset.test;

    let IMAGE_SIZE = fullset.training[0].input.length;
    let BATCH_SIZE = 100;

    let modelShape: ModelShape = {
        input: {
            units: IMAGE_SIZE,
            shape:[IMAGE_SIZE]
        },
        hidden: [
            {
                units: 9
            }
        ],
        output: {
            units: 10,
        }
    }

    let hyperParams: HyperParams = {
        training: {
            learningRate: 3.0
        },
        output: {
            activation: 'softmax'
        },
        loss: 'meanSquaredError',
        metrics: ['accuracy'],
        batchSize: 100,
        epochs: 1,
        testFrequency: 10
    }

    let model = createModel(modelShape, hyperParams);

    trainModel(hyperParams, modelShape,
               model, trainingData, testData)
        .then(() => {
            console.log('Final evaluation');
            let evalAcc = evaluateModel(modelShape,
                                        model,
                                        testData);

            let i = Math.floor(trainingData.length / hyperParams.batchSize);
            updatePlot(evalPlot, i, evalAcc * 100);

            globalModel = model;
            enableFreehandInput();
        });
}

let lossPlot = initPlot('lossPlot', 'loss', '#f45042', 'Training losses');
let accPlot = initPlot('accPlot', 'acc', '#05933e', 'Training accuracies');
let evalPlot = initPlot('evalAccPlot', 'evalAcc', '#00d8d1', 'Evaluation accuracies');
let globalModel: tf.Model = undefined;

main();



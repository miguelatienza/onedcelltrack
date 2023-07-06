const img = document.getElementById("img");
const img_rect = img.getBoundingClientRect()
const frame = document.getElementById("frame_slider");
const channel = document.getElementById("channel_selector");
const fov = document.getElementById("fov_slider");
const contrast = document.getElementById("contrast_slider");
const Experiments = document.getElementById('experiments_dropdown');
const plot = document.getElementById('plot');

noUiSlider.create(contrast, {
  start: [0, 60000],
  connect: true,
  range: {
    min: 0,
    max: 60000,
  },
  step: 1,
  format: {
    to: function (value) {
      return Math.round(value);
    },
    from: function (value) {
      return value.replace(',-', '');
    }
  },
});

function updateSliderValue(sliderId, outputId) {
  var slider = document.getElementById(sliderId);
  var output = document.getElementById(outputId);
  output.textContent = slider.value + " / " + slider.max;
}


function updateImage() {
  const body = `frame=${frame.value}&channel=${channel.value}&fov=${fov.value}&contrast=${contrast.noUiSlider.get()}`;
  fetch("/update_image", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: body,
  })
    .then((response) => response.json())
    .then((data) => {
      const imgData = data.image_data;
      const imgSrc = `data:image/jpeg;base64,${imgData}`;
      img.src = imgSrc;
    });
}

function getPosition(event) {
  const x = (event.clientX - img_rect.left) / img.width;
  const y = (event.clientY - img_rect.top) / img.height;
  fetch("/update_particle_id", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: `x=${x}&y=${y}&frame=${frame.value}`,
  })
  .then(() => {
    updateAll();
  });
  // Do whatever else you need to do with the position
}

function updateFov() {
  const body = `fov=${fov.value}`;
  fetch("/update_fov", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: body,
  })
  .then(() => {
    updateAll();
  });
}

function updateExperiment() {
  const body = `experiment=${Experiments.value}&fov=${fov.value}`;  
  fetch("/update_experiment", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: body,
  })
  //Now update max values for fov slider and frame slider
  .then((response) => response.json())
  .then((data) => {
    const maxFov = data.max_fov;
    const maxFrame = data.max_frame;
    frame.max = maxFrame;
    fov.max = maxFov
    return Promise.resolve();
  })
  .then(() => {
    updateSliderValue('fov_slider', 'fov_value');
    updateSliderValue('frame_slider', 'frame_value');
    updateAll()
  });
}

//
// Plot goes here
//
const data = [
  {x: [0, 1, 2, 3, 4, 5], y: [0, 1, 3, 2, 4, 3], name: "", line: {color: "red"}},
  {x: [0, 1, 2, 3, 4, 5], y: [0, 2, 1, 3, 2, 4], name: "", line: {color: "red"}},
  {x: [0, 1, 2, 3, 4, 5], y: [0, 1, 2, 1, 3, 2], name: "", line: {color: "green"}},
];

const layout = {
  xaxis: {
      title: "Time"
  },
  yaxis: {
      title: "Position"
  },
  margin: {
    t: 10,
    r: 10,
  },
  showlegend: false,
  height: 600,
  shapes: [
      {
          type: "rect",
          x0: 1,
          x1: 3,
          y0: 1,
          y1: 3,
          fillcolor: "#d3d3d3",
          opacity: 0.2,
          line: {
              width: 0
          }
      },
      {
          type: "rect",
          x0: 3,
          x1: 5,
          y0: 2,
          y1: 4,
          fillcolor: "#ffa07a",
          opacity: 0.2,
          line: {
              width: 0
          }
      }
  ]
};

Plotly.newPlot("plot", data, layout);

function updatePlot() {
  fetch("/update_plot", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
  })
    .then(response => response.json())
    .then(data => {
      console.log(data)
      // extract the data from the response
      const xData = data.time;
      const yDataFront = data.cell_front;
      const yDataRear = data.cell_rear;
      const yDataNucleus = data.cell_nucleus;
      const shapesData = data.shapes_data;
      
      // update the x and y data for the three lines
      Plotly.update("plot", {x: [[...xData], [...xData], [...xData]], y: [yDataFront, yDataRear, yDataNucleus]});

      // update the shapes
      const shapes = [];
      for (let i = 0; i < shapesData.length; i++) {
        const shapeData = shapesData[i];
        const shape = {
          type: "rect",
          x0: shapeData.x0,
          x1: shapeData.x1,
          y0: shapeData.y0,
          y1: shapeData.y1,
          fillcolor: shapeData.fillcolor,
          opacity: shapeData.opacity,
          line: {
            width: 0
          }
        };
        console.log(shapes)
        shapes.push(shape);
      }
      Plotly.relayout("plot", {shapes: shapes});
    });
}

// 
// update stuff
//

// contrast.noUiSlider.on("mouseup", () => {
//   updateImage();
// });

contrast.noUiSlider.on("change", () => { updateImage(); });

// function updateAll() {
//   updateImage().then(() => {
//     updatePlot();
//   });
// }
function updateAll() {
  updateImage();
  updatePlot();
  }

Experiments.onchange = updateExperiment;
frame.addEventListener("mouseup", updateAll);
channel.oninput = updateImage;
fov.addEventListener("mouseup", updateFov);

updateExperiment();
updateFov();
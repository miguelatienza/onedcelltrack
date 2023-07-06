function createPositionPlot(data) {
     var trace = [
       {x: data.t, y: data.xf, name: "", line: {color: "red"}},
       {x: data.t, y: data.xb, name: "", line: {color: "red"}},
       {x: data.t, y: data.xc, name: "", line: {color: "green"}},
     ];
    console.log(data.t)
    console.log(data.xf)
    // var trace = [
    //   {x: [0, 1, 2, 3, 4, 5], y: [0, 1, 3, 2, 4, 3], name: "", line: {color: "red"}},
    //   {x: [0, 1, 2, 3, 4, 5], y: [0, 2, 1, 3, 2, 4], name: "", line: {color: "red"}},
    //   {x: [0, 1, 2, 3, 4, 5], y: [0, 1, 2, 1, 3, 2], name: "", line: {color: "green"}},
    // ];
    
    var layout = {
      xaxis: {
        showticklabels: false,
        tickmode: 'none',
      },
      yaxis: {
        title: 'Position (um)'
      },
      margin: {
        t: 10,
        r: 10,
        b: 2,
        l: 40,
      },
      showlegend: false,
      
    };
    
    Plotly.newPlot('position-plot', trace, layout);
  }
  
  function createKappaPlot(data) {
    var trace = {
      x: data.time,
      y: data.kappa,
      type: 'scatter',
      mode: 'lines',
      name: 'kappa'
    };
    
    var layout = {
      xaxis: {
        showticklabels: false,
        tickmode: 'none',
      },
      yaxis: {
        title: 'Kappa'
      },
      margin: {
        t: 2,
        r: 10,
        b: 2,
        l: 40,
      },
      showlegend: false,
      
    };
    
    Plotly.newPlot('kappa-plot', [trace], layout);
  }
  
  function createVRPlot(data) {
    var trace = {
      x: data.time,
      y: data.vr,
      type: 'scatter',
      mode: 'lines',
      name: 'vr'
    };
    
    var layout = {
      xaxis: {
        showticklabels: false,
        tickmode: 'none',
      },
      yaxis: {
        title: 'Retrograde flow (um/s)'
      },
      margin: {
        t: 10,
        r: 10,
        b: 10,
        l: 40,
      },
      showlegend: false,
      
    };
    
    Plotly.newPlot('vr-plot', [trace], layout);
  }
  
  function createForcePlot(data) {
    var trace = {
      x: data.time,
      y: data.force,
      type: 'scatter',
      mode: 'lines',
      name: 'force'
    };
    
    var layout = {
      xaxis: {
        title: 'Time (s)'
      },
      yaxis: {
        title: 'Force (nN)'
      },
      margin: {
        t: 10,
        r: 10,
        b: 30,
        l: 40,
      },
      showlegend: false,
      
    };
    
    Plotly.newPlot('force-plot', [trace], layout);
  }

  function runSimulation() {
    // Get input values from sliders
    const E = parseFloat(document.getElementById("E").value);
    const L_0=  parseFloat(document.getElementById("L_0").value);
    const V_e0= parseFloat(document.getElementById("V_e0").value);
    const k_minus= parseFloat(document.getElementById("k_minus").value);
    const c_1= parseFloat(document.getElementById("c_1").value);
    const c_2= parseFloat(document.getElementById("c_2").value);
    const c_3= parseFloat(document.getElementById("c_3").value);
    const kappa_max= parseFloat(document.getElementById("kappa_max").value);
    const K_kappa= parseFloat(document.getElementById("K_kappa").value);
    const n_kappa= parseFloat(document.getElementById("n_kappa").value);
    const kappa_0= parseFloat(document.getElementById("kappa_0").value);
    const zeta_max= parseFloat(document.getElementById("zeta_max").value);
    const K_zeta= parseFloat(document.getElementById("K_zeta").value);
    const n_zeta= parseFloat(document.getElementById("n_zeta").value);
    const b= parseFloat(document.getElementById("b").value);
    const zeta_0= parseFloat(document.getElementById("zeta_0").value);
    const aoverN= parseFloat(document.getElementById("aoverN").value);
    const epsilon= parseFloat(document.getElementById("epsilon").value);
    const B= parseFloat(document.getElementById("B").value);

    
    // Create request body
    const params = [E, L_0, V_e0, k_minus, c_1, c_2, c_3, kappa_max, K_kappa, n_kappa, kappa_0, zeta_max, K_zeta, n_zeta, b, zeta_0, aoverN, epsilon, B]
    
    // Send POST request to run simulation and get data
    fetch("/run_simulation", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body : `params=${params}`,
    })
      .then((response) => response.json())
      .then((data) => {
        // console.log(data);
        // Call plotting functions with simulation data
        // console.log(data.t)
        createPositionPlot(data);
        createKappaPlot({ time: data.t, kappa: data.kf });
        createVRPlot({ time: data.t, vr: data.vrf });
        createForcePlot({ time: data.t, force: data.xb });
      });
  }
  
runSimulation();
// console.log('hello')
function update_position_Plot() {

  const E = parseFloat(document.getElementById("E").value);
  const L_0=  parseFloat(document.getElementById("L_0").value);
  const V_e0= parseFloat(document.getElementById("V_e0").value);
  const k_minus= parseFloat(document.getElementById("k_minus").value);
  const c_1= parseFloat(document.getElementById("c_1").value);
  const c_2= parseFloat(document.getElementById("c_2").value);
  const c_3= parseFloat(document.getElementById("c_3").value);
  const kappa_max= parseFloat(document.getElementById("kappa_max").value);
  const K_kappa= parseFloat(document.getElementById("K_kappa").value);
  const n_kappa= parseFloat(document.getElementById("n_kappa").value);
  const kappa_0= parseFloat(document.getElementById("kappa_0").value);
  const zeta_max= parseFloat(document.getElementById("zeta_max").value);
  const K_zeta= parseFloat(document.getElementById("K_zeta").value);
  const n_zeta= parseFloat(document.getElementById("n_zeta").value);
  const b= parseFloat(document.getElementById("b").value);
  const zeta_0= parseFloat(document.getElementById("zeta_0").value);
  const aoverN= parseFloat(document.getElementById("aoverN").value);
  const epsilon= parseFloat(document.getElementById("epsilon").value);
  const B= parseFloat(document.getElementById("B").value);

    
  // Create request body
  const params = [E, L_0, V_e0, k_minus, c_1, c_2, c_3, kappa_max, K_kappa, n_kappa, kappa_0, zeta_max, K_zeta, n_zeta, b, zeta_0, aoverN, epsilon, B]

  fetch("/update_simulation", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body : `params=${params}`,
  })
    .then(response => response.json())
    .then(data => {
      console.log(data)
      // extract the data from the response
      const xData = data.t;
      const yDataFront = data.xf;
      const yDataRear = data.xb;
      const yDataNucleus = data.xc;
      
      // update the x and y data for the three lines
      Plotly.update("position-plot", {x: [[...xData], [...xData], [...xData]], y: [yDataFront, yDataRear, yDataNucleus]});
    });
}

const E = document.getElementById("E");
E.onchange = update_position_Plot;
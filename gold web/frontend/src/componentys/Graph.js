import React from 'react';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const Graph = () => {
  // Manually provide data for gold carats and their corresponding prices
  const caratLabels = ['24K', '22K', '18K', '14K', '10K', '9K', '8K', '7K', '6K', '5K'];
  const priceValues = [50, 45, 40, 35, 30, 25, 20, 15, 10, 5];

  const data = {
    labels: caratLabels,
    datasets: [
      {
        label: 'Price',
        data: priceValues,
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
      },
    ],
  };

   const options = {
    indexAxis: 'x',
    elements:{
      bar:{
        borderWidth: 2,
        borderColor: 'white',
        hoverBackgroundColor: 'rgb(255, 99, 132)',
        hoverBorderColor: 'rgb(255, 99, 132)'
      }
    },
    responsive: true,
    scales: {
      x: {
        ticks: {
          color: 'white', // Set the color of x-axis ticks to white
        },
      },
      y: {
        ticks: {
          color: 'white', // Set the color of y-axis ticks to white
        },
      },
    },
    plugins: {
      legend: {
        position: 'right',
      },
      title: {
        display: true,
        text: 'Gold Carat Price Chart',
      },
    },
  };

  return (
    <div className='container'>
      <Bar className='chart-bars' data={data} options={options} />
    </div>
  );
};

export default Graph;

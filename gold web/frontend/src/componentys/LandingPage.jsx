import React from 'react'
import '../Styles/landingPage.css'
import {Link} from 'react-router-dom'
import Graph from './Graph'
const LandingPage = () => {
  return (
    <div className='main-container'>
        <h1 className='header'>Gold Price Prediction</h1>
        <div className='sub-container'>
            <Graph/> 
            
           <div className='text-container'>
              <div className='sub-con2'>
                <h2 className='sub-header'>
                Let's predict some gold prices! 
                </h2>
                <Link to='/content'>
                <button className='btn'>
                  Get Started
                </button>
                </Link>
              </div>
              </div>
        </div>
    </div>
  )
}

export default LandingPage
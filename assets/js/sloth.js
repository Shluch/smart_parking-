// sloth.js
document.addEventListener("DOMContentLoaded", function() {
  const slotsContainer = document.getElementById('slots');
  slotsContainer.innerHTML = ''; // Clear any existing content

  const totalSlots = 6;
  // Create the parking slots
  for (let i = 0; i < totalSlots; i++) {
    let slotDiv = document.createElement('div');
    // For demonstration: even-indexed slots are 'available' and odd-indexed are 'booked'
    slotDiv.className = 'slot ' + (i % 2 === 0 ? 'available' : 'booked');
    slotDiv.innerText = i + 1;

    // Add click event listener to redirect if the slot is available
    slotDiv.addEventListener('click', function() {
      if (this.classList.contains('available')) {
        window.location.href = parkUrl + "?slot=" + (i + 1);
      } else {
        alert("This slot is already booked.");
      }
    });

    slotsContainer.appendChild(slotDiv);
  }
});

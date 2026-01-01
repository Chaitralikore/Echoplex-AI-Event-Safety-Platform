


const API_URL = 'http://localhost:3000/api/attendees';
const EVENT_ID = 'TEST-EVENT-001';

async function runTest() {
    console.log('Starting Bulk API Test...');

    // 1. Setup: Create attendees
    const attendees = [
        { name: 'Test 1', email: 't1@example.com', ticketId: 'TKT-TEST-1', eventId: EVENT_ID },
        { name: 'Test 2', email: 't2@example.com', ticketId: 'TKT-TEST-2', eventId: EVENT_ID },
        { name: 'Test 3', email: 't3@example.com', ticketId: 'TKT-TEST-3', eventId: EVENT_ID },
    ];

    console.log('Registering attendees...');
    // Use bulk-import to register
    const importRes = await fetch(`${API_URL}/bulk-import`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ attendeeList: attendees, eventId: EVENT_ID })
    });
    const importData = await importRes.json();
    console.log('Import Result:', importData.success ? 'Success' : 'Failed');

    if (!importData.success) process.exit(1);

    const ticketIds = attendees.map(a => a.ticketId);

    // 2. Test Bulk Check-in
    console.log('Testing Bulk Check-in...');
    const checkInRes = await fetch(`${API_URL}/bulk-check-in`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticketIds, eventId: EVENT_ID })
    });
    const checkInData = await checkInRes.json();
    console.log('Check-in Result:', checkInData);

    if (checkInData.data.successfulCount !== 3) {
        console.error('Expected 3 successful check-ins');
    }

    // Verify status of one
    const statusRes = await fetch(`${API_URL}/status/${ticketIds[0]}/${EVENT_ID}`);
    const statusData = await statusRes.json();
    console.log(`Status of ${ticketIds[0]}:`, statusData.data.status);
    if (statusData.data.status !== 'checked_in') console.error('FAIL: Should be checked_in');

    // 3. Test Bulk Check-out
    console.log('Testing Bulk Check-out...');
    const checkOutRes = await fetch(`${API_URL}/bulk-check-out`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticketIds, eventId: EVENT_ID })
    });
    const checkOutData = await checkOutRes.json();
    console.log('Check-out Result:', checkOutData);

    if (checkOutData.data.successfulCount !== 3) {
        console.error('Expected 3 successful check-outs');
    }

    // Verify status
    const statusRes2 = await fetch(`${API_URL}/status/${ticketIds[0]}/${EVENT_ID}`);
    const statusData2 = await statusRes2.json();
    console.log(`Status of ${ticketIds[0]}:`, statusData2.data.status);
    if (statusData2.data.status !== 'checked_out') console.error('FAIL: Should be checked_out');

    // Cleanup
    console.log('Cleaning up...');
    await fetch(`${API_URL}/clear/${EVENT_ID}`, { method: 'DELETE' });
    console.log('Test Complete');
}

runTest().catch(console.error);
